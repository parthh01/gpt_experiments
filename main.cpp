#include <iostream> 
#include <string> 
#include "parser.h"
#include <vector>

using namespace std; 

int main() {
	string filepath = "data/shakespeare.txt";
	vector<char> file_arr = read_file(filepath); 
	for (char i: file_arr){
		cout << i ; 
	}; 
	return 0; 
} 


