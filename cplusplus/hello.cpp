#include <iostream>
#include <string>
using namespace std;

string greet() {
    return "Hello World";
}

int main() {
    string hello = greet();
    cout << hello << endl;
    return 0;
}