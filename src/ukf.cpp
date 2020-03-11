#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pow;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd::Identity(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 3.0;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.6;

	/**
	 * DO NOT MODIFY measurement noise values below.
	 * These are provided by the sensor manufacturer.
	 */

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	/**
	 * End DO NOT MODIFY section for measurement noise values
	 */

	/**
	 * TODO: Complete the initialization. See ukf.h for other member properties.
	 * Hint: one or more values initialized above might be wildly off...
	 */
	is_initialized_ = false;
	n_x_ = 5;                    // state dimension is set to 5 as we are going to estimate 5 parameters.
	n_aug_ = 7;                  //Augmented state dimension
	n_sig_ = 2 * n_aug_ + 1;     //number of sigma points.
	weights_ = VectorXd(n_sig_); //Weights of sigma points
	lambda_ = 3.0 - n_x_;        // Sigma point spreading parameter

	Xsig_pred_ = MatrixXd(n_x_, n_sig_);
	Xsig_pred_.fill(0.0);

	// Setup augmented weights vector
	double w0 = lambda_ / (lambda_ + n_aug_);
	double w = 1 / (2 * (lambda_ + n_aug_));
	weights_.fill(w);
	weights_(0) = w0;
}

UKF::~UKF()
{
}
void UKF::InitializeUKF(MeasurementPackage meas_package)
{
	time_us_ = meas_package.timestamp_;
	x_.fill(0.0);
	x_.head(2) << meas_package.raw_measurements_;

	is_initialized_ = true;
}

void UKF::Prediction(double delta_t)
{
	/**
	 * TODO: Complete this function! Estimate the object's location.
	 * Modify the state vector, x_. Predict sigma points, the state,
	 * and the state covariance matrix.
	 */
	// Sigma points are generates using here
	MatrixXd X_sigma = MatrixXd(n_aug_, n_sig_);
	X_sigma.fill(0.0);
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_; // Augment the mean state

	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = pow(std_a_, 2);
	P_aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2);

	MatrixXd A_aug = P_aug.llt().matrixL();
	double c_aug = sqrt(lambda_ + n_aug_); // sigma point calculation using square root
	MatrixXd cA_aug = c_aug * A_aug; // The general formula is 2N+1 so n_sig = 2*n_aug+1

	X_sigma.col(0) = x_aug;
	for (int it = 1; it <= n_aug_; it++)
	{
		X_sigma.col(it) = x_aug + cA_aug.col(it - 1);
	}
	for (int it = n_aug_ + 1; it <= n_sig_ - 1; it++)
	{
		X_sigma.col(it) = x_aug - cA_aug.col(it - 1 - n_aug_);
	}

	double dt = delta_t;
	Xsig_pred_.fill(0.0);

	for (int n_i = 0; n_i < n_sig_; n_i++)
	{
		double pos_x = X_sigma(0, n_i);
		double pos_y = X_sigma(1, n_i);
		double velocity = X_sigma(2, n_i);
		double psi = X_sigma(3, n_i);
		double psid = X_sigma(4, n_i);
		double nu_a = X_sigma(5, n_i);
		double nu_psidd = X_sigma(6, n_i);

		if (std::fabs(psid) > 0.001)
		{
			Xsig_pred_(0, n_i) = pos_x + velocity / psid * (sin(psi + psid * dt) - sin(psi))
					+ pow(dt, 2) / 2 * cos(psi) * nu_a;
			Xsig_pred_(1, n_i) = pos_y + velocity / psid * (-cos(psi + psid * dt) + cos(psi))
					+ pow(dt, 2) / 2 * sin(psi) * nu_a;
		}
		else
		{
			Xsig_pred_(0, n_i) = pos_x + velocity * dt * cos(psi) + pow(dt, 2) / 2 * cos(psi) * nu_a;
			Xsig_pred_(1, n_i) = pos_y + velocity * dt * sin(psi) + pow(dt, 2) / 2 * sin(psi) * nu_a;
		}

		Xsig_pred_(2, n_i) = velocity + 0 + dt * nu_a;
		Xsig_pred_(3, n_i) = psi + psid * dt + pow(dt, 2) / 2 * nu_psidd;
		Xsig_pred_(4, n_i) = psid + 0 + dt * nu_psidd;
	}

	VectorXd x = VectorXd(n_x_);
	x.fill(0.0);
	MatrixXd P = MatrixXd(n_x_, n_x_);
	P.fill(0.0);
    // the following code computes the weight of sigma points.
	for (int i = 0; i < n_sig_; i++)
	{
		x = x + weights_(i) * Xsig_pred_.col(i);
	}

	for (int i = 0; i < n_sig_; i++)
	{
		P = P + weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
	}

	x_ = x;
	P_ = P;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
	/**
	 * TODO: Complete this function! Make sure you switch between lidar and radar
	 * measurements.
	 */

	if (!is_initialized_)
	{

		InitializeUKF(meas_package);

		if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_)
		{
			UpdateLidar(meas_package);
		}
		else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_)
		{
			UpdateRadar(meas_package);
		}
	}
	else
	{

		double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
		time_us_ = meas_package.timestamp_;
		Prediction(delta_t);

		if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_)
		{
			UpdateLidar(meas_package);
		}
		else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_)
		{
			UpdateRadar(meas_package);
		}
	}
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
	/**
	 * TODO: Complete this function! Use lidar data to update the belief
	 * about the object's position. Modify the state vector, x_, and
	 * covariance, P_.
	 * You can also calculate the lidar NIS, if desired.
	 */

	VectorXd z = meas_package.raw_measurements_; // Reading the raw measurments
	int n_z = z.size();

	MatrixXd H = MatrixXd(n_z, n_x_); // Creating H matrix of size n_z*n_x
	H << 1, 0, 0, 0, 0, 
	     0, 1, 0, 0, 0; // Initializing H matrix with required state parameter positions.

	MatrixXd R = MatrixXd(n_z, n_z); // Creating covariance noise R matrix for
	R << pow(std_laspx_, 2), 0, 
	     0, pow(std_laspy_, 2);

	VectorXd z_pred = VectorXd(n_z);
	z_pred = x_.head(n_z);

	VectorXd y = z - z_pred;
	MatrixXd Ht = H.transpose();
	MatrixXd S = H * P_ * Ht + R;
	MatrixXd Sinv = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Sinv;

	MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

	x_ = x_ + (K * y);
	P_ = (I - K * H) * P_;
}

double angle_helper(double angle)
{
	angle = std::fmod(angle + M_PI, 2 * M_PI);  // angle in rad
	if (angle < 0)
		angle += 2 * M_PI;
	return angle - M_PI;
}


void UKF::UpdateRadar(MeasurementPackage meas_package)
{
	/**
	 * TODO: Complete this function! Use radar data to update the belief
	 * about the object's position. Modify the state vector, x_, and
	 * covariance, P_.
	 * You can also calculate the radar NIS, if desired.
	 */

	VectorXd z = meas_package.raw_measurements_;
	int n_z = z.size();

	MatrixXd Zsig = MatrixXd(n_z, n_sig_);
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	for (int i = 0; i < n_sig_; i++)
	{
		double pos_x = Xsig_pred_(0, i);
		double pos_y = Xsig_pred_(1, i);
		double velocity = Xsig_pred_(2, i);
		double psi = Xsig_pred_(3, i);
		double psid = Xsig_pred_(4, i);

		double rho = sqrt(pow(pos_x, 2) + pow(pos_y, 2));
		double phi = std::atan2(pos_y, pos_x);
		double rhod = 0.0;
		if (std::fabs(rho) > 0.001)
		{
			rhod = (pos_x * cos(psi) * velocity + pos_y * sin(psi) * velocity) / rho;
		}

		Zsig(0, i) = rho;
		Zsig(1, i) = phi;
		Zsig(2, i) = rhod;
	}

	for (int i = 0; i < n_sig_; i++)
	{
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	MatrixXd R = MatrixXd(n_z, n_z);
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);

	double std_rho2 = pow(std_radr_, 2);
	double std_phi2 = pow(std_radphi_, 2);
	double std_rhod2 = pow(std_radrd_, 2);
	double mod_angle = 0.0;

	R << std_rho2, 0, 0, 
	    0, std_phi2, 0, 
		0, 0, std_rhod2;

	for (int i = 0; i < n_sig_; i++)
	{
		VectorXd z_diff = Zsig.col(i) - z_pred;

		z_diff(1) = angle_helper(z_diff(1));

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	S = S + R;

	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);

	for (int i = 0; i < n_sig_; i++)
	{
		VectorXd z_diff = Zsig.col(i) - z_pred;
		z_diff(1) = angle_helper(z_diff(1));

		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		x_diff(3) = angle_helper(x_diff(3));

		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	MatrixXd K = MatrixXd(n_x_, n_z);
	MatrixXd S_Inverse = S.inverse();
	VectorXd residuals = z - z_pred;
	residuals(1) = angle_helper(residuals(1));

	K = Tc * S_Inverse;

	x_ = x_ + K * residuals;
	MatrixXd Kt = K.transpose();
	P_ = P_ - K * S * Kt;
}
