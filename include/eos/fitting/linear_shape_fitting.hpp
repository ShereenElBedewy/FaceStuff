/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/linear_shape_fitting.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef LINEARSHAPEFITTING_HPP_
#define LINEARSHAPEFITTING_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"

//#include "Eigen/LU"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <vector>
#include <cassert>

namespace eos {
	namespace fitting {

/**
 * Fits the shape of a Morphable Model to given 2D landmarks (i.e. estimates the maximum likelihood solution of the shape coefficients) as proposed in [1].
 * It's a linear, closed-form solution fitting of the shape, with regularisation (prior towards the mean).
 *
 * [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.
 *
 * Note: Using less than the maximum number of coefficients to fit is not thoroughly tested yet and may contain an error.
 * Note: Returns coefficients following standard normal distribution (i.e. all have similar magnitude). Why? Because we fit using the normalised basis?
 * Note: The standard deviations given should be a vector, i.e. different for each landmark. This is not implemented yet.
 *
 * @param[in] morphable_model The Morphable Model whose shape (coefficients) are estimated.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix from model to screen-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @param[in] base_face The base or reference face from where the fitting is started. Usually this would be the models mean face, which is what will be used if the parameter is not explicitly specified.
 * @param[in] lambda The regularisation parameter (weight of the prior towards the mean). Gets normalized by the number of images given.
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[in] detector_standard_deviation The standard deviation of the 2D landmarks given (e.g. of the detector used), in pixels.
 * @param[in] model_standard_deviation The standard deviation of the 3D vertex points in the 3D model, projected to 2D (so the value is in pixels).
 * @return The estimated shape-coefficients (alphas).
 */
inline std::vector<float> fit_shape_to_landmarks_linear_multi(morphablemodel::MorphableModel morphable_model, std::vector<cv::Mat> affine_camera_matrix, std::vector<std::vector<cv::Vec2f>> landmarks, std::vector<std::vector<int>> vertex_ids, std::vector<cv::Mat> base_face=std::vector<cv::Mat>(), float lambda=3.0f, boost::optional<int> num_coefficients_to_fit=boost::optional<int>(), boost::optional<float> detector_standard_deviation=boost::optional<float>(), boost::optional<float> model_standard_deviation=boost::optional<float>())
{
	using cv::Mat;
	assert(affine_camera_matrix.size() == landmarks.size() && landmarks.size() == vertex_ids.size()); // same number of instances (i.e. images/frames) for each of them

	int num_coeffs_to_fit = num_coefficients_to_fit.get_value_or(morphable_model.get_shape_model().get_num_principal_components());
	int num_images = affine_camera_matrix.size();

    // the regularisation has to be adjusted when more than one image is given
    lambda *= num_images;

	int total_num_landmarks_dimension = 0;
	for (auto&& l : landmarks) {
		total_num_landmarks_dimension += l.size();
	}

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * total_num_landmarks_dimension, num_coeffs_to_fit, CV_32FC1);
	int V_hat_h_row_index = 0;
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
	Mat P = Mat::zeros(3 * total_num_landmarks_dimension, 4 * total_num_landmarks_dimension, CV_32FC1);
	int P_index = 0;
    Mat Omega = Mat::zeros(3 * total_num_landmarks_dimension, 3 * total_num_landmarks_dimension, CV_32FC1);
    int Omega_index = 0; // this runs the same as P_index
	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * total_num_landmarks_dimension, 1, CV_32FC1);
	int y_index = 0; // also runs the same as P_index. Should rename to "running_index"?
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * total_num_landmarks_dimension, 1, CV_32FC1);
	int v_bar_index = 0; // also runs the same as P_index. But be careful, if I change it to be only 1 variable, only increment it once! :-)
						 // Well I think that would make it a bit messy since we need to increment inside the for (landmarks...) loop. Try to refactor some other way.

	for (int k = 0; k < num_images; ++k)
	{
		// For each image we have, set up the equations and add it to the matrices:
		assert(landmarks[k].size() == vertex_ids[k].size()); // has to be valid for each img
		
		int num_landmarks = static_cast<int>(landmarks[k].size());

		if (base_face[k].empty())
		{
			base_face[k] = morphable_model.get_shape_model().get_mean();
		}

		// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
		// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
		//Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			Mat basis_rows = morphable_model.get_shape_model().get_normalised_pca_basis(vertex_ids[k][i]); // In the paper, the not-normalised basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalised basis is fine/better.
			//basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
			basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(V_hat_h_row_index, V_hat_h_row_index + 3));
			V_hat_h_row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
		}
		// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
		//Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			Mat submatrix_to_replace = P.colRange(4 * P_index, (4 * P_index) + 4).rowRange(3 * P_index, (3 * P_index) + 3);
			affine_camera_matrix[k].copyTo(submatrix_to_replace);
			++P_index;
		}
		// The variances: Add the 2D and 3D standard deviations.
		// If the user doesn't provide them, we choose the following:
		// 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
		// 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
		// The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
		float sigma_squared_2D = std::pow(detector_standard_deviation.get_value_or(std::sqrt(3.0f)), 2) + std::pow(model_standard_deviation.get_value_or(0.0f), 2);
		//Mat Sigma = Mat::zeros(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
		for (int i = 0; i < 3 * num_landmarks; ++i) {
            // Sigma(i, i) = sqrt(sigma_squared_2D), but then Omega is Sigma.t() * Sigma (squares the diagonal) - so we just assign 1/sigma_squared_2D to Omega here:
            Omega.at<float>(Omega_index, Omega_index) = 1.0f / sigma_squared_2D; // the higher the sigma_squared_2D, the smaller the diagonal entries of Sigma will be
            ++Omega_index;
		}
		
		// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
		//Mat y = Mat::ones(3 * num_landmarks, 1, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			y.at<float>(3 * y_index, 0) = landmarks[k][i][0];
			y.at<float>((3 * y_index) + 1, 0) = landmarks[k][i][1];
			//y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
			++y_index;
		}
		// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
		//Mat v_bar = Mat::ones(4 * num_landmarks, 1, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			//cv::Vec4f model_mean = morphable_model.get_shape_model().get_mean_at_point(vertex_ids[i]);
			cv::Vec4f model_mean(base_face[k].at<float>(vertex_ids[k][i] * 3), base_face[k].at<float>(vertex_ids[k][i] * 3 + 1), base_face[k].at<float>(vertex_ids[k][i] * 3 + 2), 1.0f);
			v_bar.at<float>(4 * v_bar_index, 0) = model_mean[0];
			v_bar.at<float>((4 * v_bar_index) + 1, 0) = model_mean[1];
			v_bar.at<float>((4 * v_bar_index) + 2, 0) = model_mean[2];
			//v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
			++v_bar_index;
			// note: now that a Vec4f is returned, we could use copyTo?
		}
	}

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega
	Mat A = P * V_hat_h; // camera matrix times the basis
	Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
	//Mat c_s; // The x, we solve for this! (the variance-normalised shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
	//int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
	const int num_shape_pc = num_coeffs_to_fit;
	Mat AtOmegaA = A.t() * Omega * A;
	Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(num_shape_pc, num_shape_pc, CV_32FC1);

	// Invert (and perform some sanity checks) using Eigen:
/*	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> AtOmegaAReg_Eigen(AtOmegaAReg.ptr<float>(), AtOmegaAReg.rows, AtOmegaAReg.cols);
	Eigen::FullPivLU<RowMajorMatrixXf> luOfAtOmegaAReg(AtOmegaAReg_Eigen); // Calculate the full-pivoting LU decomposition of the regularized AtA. Note: We could also try FullPivHouseholderQR if our system is non-minimal (i.e. there are more constraints than unknowns).
	auto rankOfAtOmegaAReg = luOfAtOmegaAReg.rank();
	bool isAtOmegaARegInvertible = luOfAtOmegaAReg.isInvertible();
	float threshold = std::abs(luOfAtOmegaAReg.maxPivot()) * luOfAtOmegaAReg.threshold(); // originaly "2 * ..." but I commented it out
	RowMajorMatrixXf AtARegInv_EigenFullLU = luOfAtOmegaAReg.inverse(); // Note: We should use ::solve() instead
	Mat AtOmegaARegInvFullLU(AtARegInv_EigenFullLU.rows(), AtARegInv_EigenFullLU.cols(), CV_32FC1, AtARegInv_EigenFullLU.data()); // create an OpenCV Mat header for the Eigen data
*/
	// Solve using OpenCV:
	Mat c_s; // Note/Todo: We get coefficients ~ N(0, sigma) I think. They are not multiplied with the eigenvalues.
	bool non_singular = cv::solve(AtOmegaAReg, -A.t() * Omega.t() * b, c_s, cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
	// Because we're using SVD, non_singular will always be true. If we were to use e.g. Cholesky, we could return an expected<T>.

/*	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> P_Eigen(P.ptr<float>(), P.rows, P.cols);
	Eigen::Map<RowMajorMatrixXf> V_hat_h_Eigen(V_hat_h.ptr<float>(), V_hat_h.rows, V_hat_h.cols);
	Eigen::Map<RowMajorMatrixXf> v_bar_Eigen(v_bar.ptr<float>(), v_bar.rows, v_bar.cols);
	Eigen::Map<RowMajorMatrixXf> y_Eigen(y.ptr<float>(), y.rows, y.cols);
	Eigen::MatrixXf A_Eigen = P_Eigen * V_hat_h_Eigen;
	Eigen::MatrixXf res = (A_Eigen.transpose() * A_Eigen + lambda * Eigen::MatrixXf::Identity(num_coeffs_to_fit, num_coeffs_to_fit)).householderQr().solve(-A_Eigen.transpose() * (P_Eigen * v_bar_Eigen - y_Eigen));
	Mat c_s(res.rows(), res.cols(), CV_32FC1, res.data()); // Note: Could also use LDLT.
*/
	return std::vector<float>(c_s);
};

/**
 * Fits the shape of a Morphable Model to given 2D landmarks (i.e. estimates the maximum likelihood solution of the shape coefficients) as proposed in [1].
 * It's a linear, closed-form solution fitting of the shape, with regularisation (prior towards the mean).
 *
 * [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.
 *
 * Note: Using less than the maximum number of coefficients to fit is not thoroughly tested yet and may contain an error.
 * Note: Returns coefficients following standard normal distribution (i.e. all have similar magnitude). Why? Because we fit using the normalised basis?
 * Note: The standard deviations given should be a vector, i.e. different for each landmark. This is not implemented yet.
 *
 * @param[in] morphable_model The Morphable Model whose shape (coefficients) are estimated.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix from model to screen-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @param[in] base_face The base or reference face from where the fitting is started. Usually this would be the models mean face, which is what will be used if the parameter is not explicitly specified.
 * @param[in] lambda The regularisation parameter (weight of the prior towards the mean).
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[in] detector_standard_deviation The standard deviation of the 2D landmarks given (e.g. of the detector used), in pixels.
 * @param[in] model_standard_deviation The standard deviation of the 3D vertex points in the 3D model, projected to 2D (so the value is in pixels).
 * @return The estimated shape-coefficients (alphas).
 */
inline std::vector<float> fit_shape_to_landmarks_linear(const morphablemodel::MorphableModel& morphable_model, cv::Mat affine_camera_matrix, std::vector<cv::Vec2f> landmarks, std::vector<int> vertex_ids, cv::Mat base_face=cv::Mat(), float lambda=3.0f, boost::optional<int> num_coefficients_to_fit=boost::optional<int>(), boost::optional<float> detector_standard_deviation=boost::optional<float>(), boost::optional<float> model_standard_deviation=boost::optional<float>())
{
	return fit_shape_to_landmarks_linear_multi(morphable_model, { affine_camera_matrix }, { landmarks }, { vertex_ids }, { base_face }, lambda, num_coefficients_to_fit, detector_standard_deviation, model_standard_deviation );
}


	} /* namespace fitting */
} /* namespace eos */

#endif /* LINEARSHAPEFITTING_HPP_ */
