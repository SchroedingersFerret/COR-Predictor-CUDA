/*
 * COR-main.cpp
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
//   COR-Predictor is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with COR-Predictor-CUDA.  If not, see <https://www.gnu.org/licenses/>.

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "COR-predictor.h"

int COR_predictor::n_initial;
int COR_predictor::n_gpool;
int COR_predictor::n_repro;
float COR_predictor::pm;
int COR_predictor::n_elite;
int COR_predictor::n_normal;
int COR_predictor::n_data;
float COR_predictor::error;
std::vector<std::vector<float> > COR_predictor::independent;
std::vector<float> COR_predictor::dependent;
bool COR_predictor::random_parameters = false;
bool COR_predictor::quit_cor = false;
std::vector<std::vector<float> > COR_predictor::parameters_global(4,std::vector<float> (6));

int main()
{
	srand((unsigned int)time(NULL));
	COR_predictor::Get_settings();
	COR_predictor::Get_independent();
	COR_predictor::Get_dependent();
	COR_predictor::Get_parameters();
	std::cout << "Welcome to COR Predictor CUDA 0.1-alpha\n";
	std::cout << "Copyright 2019, J. Ball (SchroedingersFerret)\n\n";
	while (!COR_predictor::quit_cor)
		COR_predictor::Main_menu();
	return 0;
}
