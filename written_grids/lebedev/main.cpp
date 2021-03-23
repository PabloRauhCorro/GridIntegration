#include <iostream>
#include <vector>
#include "sphere_lebedev_rule.hpp"
#include "sphere_lebedev_rule.cpp"

void print(double* x, double* y, double*z, double* w, int size){
    double pi = 3.14159265359;
    for(int i = 0; i<size; i++){
        std::cout<<x[i] << " " <<y[i]<< " "<<z[i]<< " "<<w[i] <<std::endl;
    }
}
int main(){
    /* the variable rule can be modified*/
    int rule = 4;
    int precision = precision_table(rule);
    int order = order_table(rule);
    double x[order];
    double y[order];
    double z[order];
    double w[order];
    ld_by_order(order, x, y, z, w);
    std::cout<< "Lebedev rule number: " << rule << std::endl;
    std::cout<< "Precision of degree: " << precision << std::endl;
    std::cout<< "Number of points: " << order << std::endl << std::endl;
    print(x, y, z, w, order);
    return 1;
}

