/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <math.h>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	std::default_random_engine gen;
	// std::random_device rd;
	// std::mt19937 gen(rd());
	std::normal_distribution<double> N_x_particle(0, std[0]);
	std::normal_distribution<double> N_y_particle(0, std[1]);
	std::normal_distribution<double> N_theta_particle(0, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = x + N_x_particle(gen);
		p.y = y + N_y_particle(gen);
		p.theta = theta + N_theta_particle(gen);
		p.weight = 1.0;
		
		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;
	// std::random_device rd;
	// std::mt19937 gen(rd());
	std::normal_distribution<double> N_x_particle(0, std_pos[0]);
	std::normal_distribution<double> N_y_particle(0, std_pos[1]);
	std::normal_distribution<double> N_theta_particle(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle& p = particles[i];

		if (fabs(yaw_rate) > 0.0001) {
			const double v_over_yawd = velocity / yaw_rate;
			const double yaw_diff = yaw_rate * delta_t;

			p.x += v_over_yawd * (std::sin(p.theta + yaw_diff) - std::sin(p.theta));
			p.y += v_over_yawd * (std::cos(p.theta) - std::cos(p.theta + yaw_diff));
			p.theta += yaw_diff;
			
		} else {
			const double v_dt = velocity * delta_t;

			p.x += v_dt * std::cos(p.theta);
			p.y += v_dt * std::sin(p.theta);
		}

		p.x += N_x_particle(gen);
		p.y += N_y_particle(gen);
		p.theta += N_theta_particle(gen);
	}
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> eligible, std::vector<LandmarkObs> observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	std::vector<LandmarkObs> matched;

	for (int i = 0; i < eligible.size(); ++i) {
		double min_dist = std::numeric_limits<double>::max();
		int min_idx = 0;

		for (int j = 0; j < observations.size(); ++j) {
			double x_norm = (observations[j].x - eligible[i].x_f)*(observations[j].x - eligible[i].x_f);
			double y_norm = (observations[j].y - eligible[i].y_f)*(observations[j].y - eligible[i].y_f);
			double dist = std::sqrt(x_norm + y_norm); 

			if (dist < min_dist) {
				min_dist = dist;
				min_idx = j;
			}
		}

		LandmarkObs obs = observations[min_idx];
		obs.id = i;
		matched.push_back(obs);
		// observations.erase(observations.begin() + min_idx);
	} 
	return matched;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	double tot_weight = 0.0;

	for (int i = 0; i < num_particles; ++i) {
		Particle& p = particles[i];

		std::vector<LandmarkObs> t_observations;
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs obs = observations[j];
			double newx = p.x + obs.x*std::cos(p.theta) - obs.y*std::sin(p.theta);
			double newy = p.y + obs.x*std::sin(p.theta) + obs.y*std::cos(p.theta);
			obs.x = newx;
			obs.y = newy;
			t_observations.push_back(obs);
		}

		std::vector<Map::single_landmark_s> eligible_landmarks;
		for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
			Map::single_landmark_s lm = map_landmarks.landmark_list[k];
			double dist = std::sqrt((p.x - lm.x_f)*(p.x - lm.x_f) + (p.y - lm.y_f)*(p.y - lm.y_f));
			if (dist <= sensor_range) {
				eligible_landmarks.push_back(lm);
			}
		}
		
		std::vector<LandmarkObs> matched_landmarks = dataAssociation(eligible_landmarks, t_observations);
		double weight = 1.0;
		for (int l = 0; l < eligible_landmarks.size(); ++l) {
			Map::single_landmark_s lm = eligible_landmarks[l];
			LandmarkObs obs = matched_landmarks[l];

			double z = (obs.x - lm.x_f)*(obs.x - lm.x_f)/(std_x*std_x) + (obs.y - lm.y_f)*(obs.y - lm.y_f)/(std_y*std_y);
			double prob = 1.0 / (2.0*M_PI*std_x*std_y) * exp(-0.5 * z);

			weight *= prob;
		}

		p.weight = weight;
		tot_weight += p.weight;
	}

	for (int i = 0; i < num_particles; ++i) {
		Particle& p = particles[i];
		p.weight /= tot_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> new_particles;
	double beta = 0.0;
	double max_weight = 0.0;

	for (int i = 0; i < num_particles; ++i) {
		if (particles[i].weight > max_weight) {
			max_weight = particles[i].weight;
		}
	}

	std::default_random_engine gen;
	// std::random_device rd;
	// std::mt19937 gen(rd());
	std::uniform_int_distribution<int> U_i(0, num_particles-1);
	std::uniform_real_distribution<double> U_w(0, 2*max_weight);

	int idx = U_i(gen);
	for (int j = 0; j < num_particles; ++j) {
		beta += U_w(gen);
		while (beta > particles[idx].weight) {
			beta -= particles[idx].weight;
			idx = (idx + 1) % num_particles;
		}
		new_particles.push_back(particles[idx]);
	}

	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
