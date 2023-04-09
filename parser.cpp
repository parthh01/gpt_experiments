#include <iostream>
#include <string> 
#include <vector>

using namespace std; 



vector<char> read_file(string filename){
	int length; 
	ifstream t; 
	t.open(filename); 
	t.seekg(0,ios::end); 
	length = t.tellg();
	t.seekg(0, ios::beg);
	vector<char> buffer(length," ");
	t.read(buffer, length);
	t.close();  
	return buffer; 
};


