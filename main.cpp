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
