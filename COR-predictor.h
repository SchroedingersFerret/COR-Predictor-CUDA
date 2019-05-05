/*
 * COR-predictor.h
 *
 *  Copyright 2019
 *      J. Ball (SchroedingersFerret)
 */

//This file is part of COR-Predictor-CUDA.
//
//   COR-Predictor-CUDA is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   COR-Predictor-CUDA is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with COR-Predictor-CUDA.  If not, see <https://www.gnu.org/licenses/>.

#ifndef COR_PREDICTOR_H_
#define COR_PREDICTOR_H_

#include <vector>

class COR_predictor
{
	private:
		static std::vector<float> read_csv1d(const char * filename);
		static std::vector<std::vector<float> > read_csv2d(const char * filename);
		static void write_csv1d(std::vector<float> a, const char * filename);
		static void write_csv2d(std::vector<std::vector<float> > a, const char * filename);
		static void Point_entry();
		static bool Set_more(char input);
		static bool Enter_more();
		static bool Set_quit(char input);
		static bool Return_quit();
		static void Enter();
		static bool Set_random(char input);
		static bool Use_random();
		static bool Set_write(char input);
		static void Print_parameters(std::vector<std::vector<float> > param);
		static void Write_parameters(std::vector<std::vector<float> > param);
		static void Show_time(int time);
		static void Optimize();
		static std::vector<float> pGet_independent();
		static void Predict();
		static void Show_main_menu();
		static int Set_mode(char input);

	public:
		//initial population size
		static int n_initial;
		//size of gene pool for selection
		static int n_gpool;
		//number of chromosomes to be selected for reproduction
		static int n_repro;
		//number of independent variables
		static const int nx = 7;
		//percentage selected for mutation
		static float pm;
		//number of elites
		static int n_elite;
		//remainder of population
		static int n_normal;
		//number of datapoints
		static int n_data;
		//least squares error tolerance
		static float error;
		//independent variables array size n_data*nx
		static std::vector<std::vector<float> > independent;
		//dependent variable array size n_data
		static std::vector<float> dependent;
		//determines whether initial population contains entirely random parameters
		static bool random_parameters;
		//quits the program if true
		static bool quit_cor;
		//parameter array
		static std::vector<std::vector<float> > parameters_global;

		static void Get_settings();
		static void Get_independent();
		static void Get_dependent();
		static void Get_parameters();
		static void Main_menu();
};

#endif /* COR_PREDICTOR_H_ */
