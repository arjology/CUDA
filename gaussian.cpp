// C++ prgroam to generate Gaussian filter 
#include <cmath> 
#include <iomanip> 
#include <iostream> 
using namespace std; 
  
// Function to create Gaussian filter 
void FilterCreation(double GKernel[][15], int dim) 
{ 
    // intialising standard deviation to 1.0 
    double sigma = 4.0; 
    double r, s = 2.0 * sigma * sigma; 
  
    // sum is for normalization 
    double sum = 0.0; 
  
    // generating dim x dim kernel
    int filterWidth = dim / 2;
    for (int x = -filterWidth; x <= filterWidth; x++) { 
        for (int y = -filterWidth; y <= filterWidth; y++) { 
            r = sqrt(x * x + y * y); 
            GKernel[x + filterWidth][y + filterWidth] = (exp(-(r * r) / s)) / (M_PI * s); 
            sum += GKernel[x + filterWidth][y + filterWidth]; 
        } 
    } 
  
    // normalising the Kernel 
    for (int i = 0; i < dim; ++i) 
        for (int j = 0; j < dim; ++j) 
            GKernel[i][j] /= sum; 
} 
  
// Driver program to test above function 
int main() 
{ 
    int dim = 15;
    double GKernel[15][15]; 
    FilterCreation(GKernel, dim); 
  
    for (int i = 0; i < dim; ++i) { 
        for (int j = 0; j < dim; ++j) 
            cout << GKernel[i][j] << "\t"; 
        cout << endl; 
    } 
}
