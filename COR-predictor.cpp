/*
 * COR-predictor.cpp
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

//reads and applies settings
void COR_predictor::Get_settings()
{
	std::ifstream fin;
	fin.open("settings.txt");
	if (fin.fail())
	{
		std::cout << "Error: File 'settings.txt' not found.\n";
		abort();
	}
	char ch;
	
	do
	{
		fin.get(ch);
	}while(ch!='=');
	fin >> n_initial;

	do
	{
		fin >> ch;
	}while(ch!='=');
	fin >> n_gpool;
	n_repro = 0.5*n_gpool;

	do
	{
		fin >> ch;
	}while(ch!='=');
	fin >> n_elite;
	n_normal = n_gpool-n_elite;
	
	do
	{
		fin >> ch;
	}while(ch!='=');
	fin >> pm;

	do
	{
		fin >> ch;
	}while(ch!='=');
	fin >> error;

	fin.close();
}

//reads 1d csv files
std::vector<float> COR_predictor::read_csv1d(const char * filename)
{
	std::ifstream fin;
	float input;
	std::vector<float> output;
	fin.open(filename);
	while(!fin.eof())
	{
		if (fin.peek()==',')
		{
			char ch;
			fin.get(ch);
		}
		if (fin >> input)
			output.push_back((float)input);
	}
	fin.close();
	return output;
}

//reads 2d csv files
std::vector<std::vector<float> > COR_predictor::read_csv2d(const char * filename)
{
	std::ifstream fin;
	float input;
	std::vector<float> datapoint;
	std::vector<std::vector<float> > output;
	fin.open(filename);
	while(!fin.eof())
	{
		
		if (fin.peek() == ',' || fin.peek() == '\n')
		{
			char ch;
			fin.get(ch);
			if (ch == '\n')
			{
				output.push_back(datapoint);
				datapoint.clear();
			}
		}
		if (fin >> input)
		{
			datapoint.push_back((float)input);
		}
	}
	fin.close();
	return output;
}

//writes 1d vector to csv file
void COR_predictor::write_csv1d(std::vector<float> a, const char * filename)
{
	std::ofstream fout;
	fout.open(filename);
	fout.precision(30);
	fout.setf(std::ios::fixed, std::ios::floatfield);
	int ni = a.size();
	for (int i=0; i<ni; ++i)
		fout << a[i] << "\n";
	fout.close();
}

//writes 2d vector to csv file
void COR_predictor::write_csv2d(std::vector<std::vector<float> > a, const char * filename)
{
	std::ofstream fout;
	fout.open(filename);
	fout.precision(30);
	fout.setf(std::ios::fixed, std::ios::floatfield);
	const int ni = a.size();
	const int nj = a[0].size();
	for (int i=0; i<ni; ++i)
	{
		for (int j=0; j<nj; ++j)
		{
			fout << std::scientific << a[i][j];
			if (j != nj-1)
				fout << ",";
		}
		fout << "\n";
	}
	fout.close();
}

//reads the independent variables of the training datapoints
void COR_predictor::Get_independent()
{
	std::ifstream fin;
	fin.open("cor_independent.csv");
	fin.close();
	if (fin.fail())
	{
		std::cout << "Error: File 'cor_independent.csv' not found.\n";
		abort();
	}
	
	independent = read_csv2d("cor_independent.csv");
	
	if (independent[0].size() != 7)
	{
		std::cout << "Error: File 'cor_independent.csv' must be of dimension n*7.\n";
		abort();
	}
	n_data = independent.size();
}
	
//reads the dependent variables of the training datapoints
void COR_predictor::Get_dependent()
{
	std::ifstream fin;
	fin.open("cor_dependent.csv");
	fin.close();
	if (fin.fail())
	{
		std::cout << "Error: File 'cor_dependent.csv' not found.\n";
		abort();
	}
	
	dependent = read_csv1d("cor_dependent.csv");

	if (dependent.size() != independent.size())
	{
		std::cout << "Error: Files 'cor_independent.csv' and 'cor_dependent.csv' must have the same number of entries.\n";
		abort();
	}
	
	for (int i=0; i<n_data; ++i)
	{
		if (dependent[i]<0.f || dependent[i]>1.f)
		{
			std::cout << "Error: The dependent variables must be between 0.0 and 1.0.\n";
			abort();
		}
	}
}

//reads parameter array from file
void COR_predictor::Get_parameters()
{
	std::ifstream fin;
	fin.open("cor_parameters.csv");
	fin.close();
	if (!fin.fail())
	{
		std::vector<std::vector<float> > temp = read_csv2d("cor_parameters.csv");
		if (temp.size() != parameters_global.size() || temp[0].size() != parameters_global[0].size())
		{
			std::cout << "Error: File 'cor_parameters.csv' must be of dimensions " << parameters_global.size() << "*" << parameters_global[0].size() << ".\n";
			abort();
		}
		parameters_global = temp;	
	}
}

//user enters new training datapoints
void COR_predictor::Point_entry()
{
	float input;
	std::vector<float> new_point(nx);
	std::cout << "Enter the yield strength of the first object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[0] = input;
	
	std::cout << "Enter the yield strength of the second object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[1] = input;
	
	std::cout << "Enter the Young's modulus of the first object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[2] = input;
	
	std::cout << "Enter the Young's modulus of the second object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[3] = input;
	
	std::cout << "Enter the density of the first object.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[4] = input;
	
	std::cout << "Enter the density of the second object.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[5] = input;
	
	std::cout << "Enter the objects' relative velocity in m/s.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	new_point[6] = input;
	
	independent.push_back(new_point);
	
	std::cout << "Enter the coefficient of restitution.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	
	dependent.push_back(input);
}

//sets bool more
bool COR_predictor::Set_more(char input)
{
	switch(input)
	{
		case 'y':
		case 'Y':	return true;

		case 'n':
		case 'N':	return false;

		default	:	throw "Invalid input\nEnter y/n: ";
	}
}

//asks if user wants to enter more points
bool COR_predictor::Enter_more()
{
	std::cout << "Enter more datapoints?\nEnter y/n: ";
	char input;
	bool more = false;
	while(std::cin.get(input))
	{
		char dummy[30];
		std::cin.getline(dummy,30);
		std::cout << "\n";
		try
		{
			more = Set_more(input);
		}
		catch(const char * s)
		{
			std::cout << s;
			continue;
		}
		break;
	}
	std::cout << "\n";
	return more;
}

//sets bool quit
bool COR_predictor::Set_quit(char input)
{
	switch(input)
	{
		case '1':	return false;

		case '2':	return true;

		default	:	throw "Invalid input\nEnter 1/2: ";
	}
}

//asks user to return to main menu or quit program
bool COR_predictor::Return_quit()
{
	char input;
	std::cout << "Enter '1' to return to the main menu. Enter '2' to quit.\n";
	bool quit = false;
	while(std::cin.get(input))
	{
		char dummy[30];
		std::cin.getline(dummy,30);
		std::cout << "\n";
		try
		{
			quit = Set_quit(input);
		}
		catch(const char * s)
		{
			std::cout << s;
			continue;
		}
		break;
	}
	std::cout << "\n";
	return quit;
}
	
//training datapoint entry
void COR_predictor::Enter()
{
	bool enter_point = true;
	while (enter_point)
	{
		Point_entry();
		write_csv2d(independent,"cor_independent.csv");
		write_csv1d(dependent,"cor_dependent.csv");
		enter_point = Enter_more();
	}
	
	quit_cor = Return_quit();
}

//sets bool random
bool COR_predictor::Set_random(char input)
{
	switch(input)
	{
		case 'y':
		case 'Y':	return false;

		case 'n':
		case 'N':	std::cout << "Initiating with random values. (Convergence will take longer)\n\n";
					return true;

		default	:	throw "Invalid input\nEnter y/n: ";
	}
}

//returns a boolean operator to instruct program whether to randomize elite population
bool COR_predictor::Use_random()
{
	
	std::ifstream fin;
	fin.open("cor_parameters.csv");
	fin.close();
	if (fin.fail())
	{
		std::cout << "File: 'cor_parameters.csv' not found.\n";
		std::cout << "Initiating with random values. (Convergence will take longer)\n\n";
		return true;
	}
	
	std::cout << "File: 'cor_parameters.csv' found.\n\n";
	bool random = true;
	std::cout << "Initiate with these values?\nEnter y/n: ";
	char input;
	
	while(std::cin.get(input))
	{
		char dummy[30];
		std::cin.getline(dummy,30);
		std::cout << "\n";
		try
		{
			random = Set_random(input);
		}
		catch(const char * s)
		{
			std::cout << s;
			continue;
		}
		break;
	}
	return random;
}

//prints the parameters in the terminal
void COR_predictor::Print_parameters(std::vector<std::vector<float> > param)
{
	int ni = param.size();
	int nj = param[0].size();
	for (int i=0; i<ni; ++i)
	{
		for (int j=0; j<nj; ++j)
		{
			std::cout << param[i][j] << ",";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

//sets bool write
bool COR_predictor::Set_write(char input)
{
	switch(input)
	{
		case 'y':
		case 'Y':	return true;

		case 'n':
		case 'N':	return false;

		default	:	throw "\nInvalid input\nEnter y/n: ";
	}
}
	
//runs Set_write() and writes parameters to file depending on user input
void COR_predictor::Write_parameters(std::vector<std::vector<float> > param)
{
	bool write = false;
	std::cout << "Write these parameters to 'cor_parameters.csv'?\nPrevious values will be overwritten.\nEnter y/n: ";
	char input;
	
	while(std::cin.get(input))
	{
		char dummy[30];
		std::cin.getline(dummy,30);
		std::cout << "\n";
		try
		{
			write = Set_write(input);
		}
		catch(const char * s)
		{
			std::cout << s;
			continue;
		}
		break;
	}
	
	if (write)
	{
		write_csv2d(param,"cor_parameters.csv");
		std::cout << "Parameters written.\n\n";
	}
	else
		std::cout << "Optimization terminated without writing new parameters.\n\n";
}

//shows time taken for optimization
void COR_predictor::Show_time(int time)
{
	int hours, minutes, seconds;
	seconds = time%60;
	hours = time/3600;
	minutes = time/60-hours*60;
	std::cout << "Execution time: " << hours << ":";
	if (minutes<10)
		std::cout << 0;
	std::cout << minutes << ":";
	if (seconds<10)
		std::cout << 0;
	std::cout << seconds << "\n\n";
}

//optimizes parameters to training data
void COR_predictor::Optimize()
{
	random_parameters = Use_random();
	clock_t tStart = clock();
	genetic::run();
	std::cout << "\n\nParameters found:\n\n" ;
	Print_parameters(parameters_global);
	Show_time((clock()-tStart)/CLOCKS_PER_SEC);
	Write_parameters(parameters_global);
	quit_cor = Return_quit();
}

//gets independent variables for prediction
std::vector<float> COR_predictor::pGet_independent()
{
	std::vector<float> x(nx);
	float input;
	std::cout << "Enter the yield strength of the first object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[0] = input;
	
	std::cout << "Enter the yield strength of the second object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[1] = input;
	
	std::cout << "Enter the Young's modulus of the first object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[2] = input;
	
	std::cout << "Enter the Young's modulus of the second object in GPa.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[3] = input;
	
	std::cout << "Enter the density of the first object.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[4] = input;
	
	std::cout << "Enter the density of the second object.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[5] = input;
	
	std::cout << "Enter the objects' relative velocity in m/s.\n";
	std::cin >> input;
	std::cin.get();
	std::cout << "\n";
	x[6] = input;
	
	return x;
}	

//predicts a coefficient of restitution
void COR_predictor::Predict()
{
	thrust::host_vector<float> x(n_data*nx);
	flatten2dToHost(x,independent);
	thrust::host_vector<float> param(parameters_global.size()*parameters_global[0].size());
	flatten2dToHost(param,parameters_global);
	float pred_y = optimization::f(convertToKernel(x,n_data,nx,1),0,
		convertToKernel(param,parameters_global.size(),parameters_global[0].size(),1),0);
	std::cout.precision(3);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout << "e = " << pred_y << "\n\n";
	quit_cor = Return_quit();
}	

//shows the main menu
void COR_predictor::Show_main_menu()
{
	std::cout << "Enter '1' to enter a new training datapoint.\n\n";
	std::cout << "Enter '2' to optimize the parameters.\n\n";
	std::cout << "Enter '3' to predict a coefficient of restitution.\n\n";
	std::cout << "Enter '4' to quit.\n\n";
}

//sets bool quit
int COR_predictor::Set_mode(char input)
{
	switch(input)
	{
		case '1' :  return 1;

		case '2' : 	return 2;

		case '3' : 	return 3;

		case '4' : 	return 4;

		default : 	throw "Invalid input. Please enter '1','2','3', or '4'.\n\n";
	}
}	

//user can select mode of operation from the main menu
void COR_predictor::Main_menu()
{
	Show_main_menu();
	int mode;
	char input;
	while(std::cin.get(input))
	{
		char dummy[30];
		std::cin.getline(dummy,30);
		std::cout << "\n";
		try
		{
			mode = Set_mode(input);
		}
		catch(const char * s)
		{
			std::cout << s;
			continue;
		}
		break;
	}
		
	switch(mode)
	{
		case 1 	:  	Enter();
					break;
		case 2  :	Optimize();
					break;
		case 3  :  	Predict();
					break;
		case 4  :  	quit_cor = true;
					break;	
		default : 	break;
	}
}

