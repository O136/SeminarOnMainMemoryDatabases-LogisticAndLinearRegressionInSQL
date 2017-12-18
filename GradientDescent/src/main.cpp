#include <iostream>
#include <vector>
#include <armadillo>
#include <list>
#include "CSVIterator.h"

arma::mat logistic_func(arma::colvec const &theta, arma::mat const &X) {
	double const e = 2.7182818284;

//	std::cout << X << std::endl
//			  << theta << std::endl;

	arma::mat m = -1 * (X * theta);
	std::cout << X << std::endl
			  << "m matrix" << std::endl
			  << m << std::endl;
	for(int i = 0; i < m.n_rows; ++i) {
		for(int j = 0; j < m.n_cols; ++j) {
			m(i, j) = std::pow(e, m(i, j));
		}
	}

	std::cout << X << std::endl
			  << "m matrix" << std::endl
			  << m << std::endl;

	auto res = 1.0/(1 + m);
	std::cout << res << std::endl;
	return res;
}

arma::mat log_gradient(arma::colvec const &theta, arma::mat const &X, arma::mat const &y) {
	auto res = logistic_func(theta, X);
//	std::cout << "res = " << res << std::endl;
//	std::cout << "y = " << y << std::endl;
	arma::mat a = res - y;
//	std::cout << "y" << std::endl << y << std::endl
//		      << "a.t()" << std::endl << a.t() << std::endl;
//	std::cout << "x" << std::endl << X << std::endl;
//
//	std::cout << a.n_rows << " " << X.n_rows << " & "
//			  << a.n_cols << " " << X.n_cols << std::endl;



//	return arma::dot(X, a);
	return a.t() * X;
}

arma::mat gradient_descent(arma::mat const &X, arma::mat const &y, arma::colvec const &theta)
{
	//gradient descent constants
	const double alpha = 0.00001;
	const int iterations = 100000;
	arma::colvec th(theta);
	for(int i=0; i<iterations; ++i) {
//		theta = theta - alpha*((double)1/(X.n_rows))*X.t()*(X*theta-y);
		auto grad = log_gradient(theta, X, y);
//		std::cout << theta << " ____ " << grad << std::endl;
		th = th - alpha * grad.t();
	}

	return th;
}


arma::colvec fastLM(const arma::vec & y, const arma::mat & X) { //List
	int n = X.n_rows, k = X.n_cols;

	arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
	arma::colvec res  = y - X*coef;           // residuals

	// std.errors of coefficients
	double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(n - k);

	arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));

//	return List::create(Named("coefficients") = coef,
//						Named("stderr")       = std_err,
//						Named("df.residual")  = n - k);

	return coef;
}


const int SEED = 1111;

int main() {
	using namespace arma;

	/**
	 * Initialize the data
	 */

	arma::mat X(30, 2);
	arma::vec Y(30);


	/**
	 * read the data from the .csv
	 */

	std::ifstream waiter_file("../data/waiter.csv");
	std::ifstream ozone_file("../data/ozone.csv");
	std::ifstream cars_file("../data/cars.csv");

	//
	bool has_header = true;
	int i = 0;
	for(CSVIterator loop(waiter_file); loop != CSVIterator(); ++loop) {
		if(has_header) {
			has_header = false;
			continue;
		}

		double tip_val = std::stod((*loop)[0]);
		double meal_cost_val = std::stod((*loop)[1]);

		// solving for y = b + mx
		X(i, 0) = 1;
		X(i, 1) = meal_cost_val;
		Y(i) = tip_val;
		++i;
	}

	cout << "solving for y = b + mx" << std::endl
		 << fastLM(Y, X) << std::endl;

	/**
	 * resetting the data
	 */
	i = 0;
	has_header = true;
	X.resize(330, 4);
	Y.resize(330);

	for(CSVIterator loop(ozone_file); loop != CSVIterator(); ++loop) {
		if(has_header) {
			has_header = false;
			continue;
		}

		double ozone = std::stod((*loop)[0]);
		double sea_level = std::stod((*loop)[1]);
		double humidity = std::stod((*loop)[3]);
		double temp =  std::stod((*loop)[4]);

		// solving for y = b + mx
		X(i, 0) = 1;
		X(i, 1) = humidity;
		X(i, 2) = temp;
		X(i, 3) = sea_level;
		Y(i) = ozone;
		++i;
	}


	cout << "solving for ozone = b + sea_level*x + humidity*x + temp*x" << std::endl
		 << fastLM(Y, X) << std::endl;

	/**
	 * logistic regression with gradient descent
 	 * resetting the data
 	 */
	i = 0;
	has_header = true;
	X.resize(32, 3);
	Y.resize(32);
	arma::colvec theta {10., -2.9, 2.1};


	for(CSVIterator loop(cars_file); loop != CSVIterator(); ++loop) {
		if(has_header) {
			has_header = false;
			continue;
		}

		double cylinders = std::stod((*loop)[2]);
		double weight = std::stod((*loop)[6]);
		double vertical = std::stod((*loop)[8]);

		// solving for y = b + mx
		X(i, 0) = 1;
		X(i, 1) = cylinders;
		X(i, 2) = weight;
		Y(i) = vertical;
		++i;
	}

	cout << "logistic regression parameters"
		 << gradient_descent(X, Y, theta)
		 << std::endl;


	return EXIT_SUCCESS;
}