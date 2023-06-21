// shell command wrapper

#include <iostream>
#include <vector>
#include "clAmpcor.h"


int main() {
    cl::Ampcor::Ampcor ampcor;
    ampcor.run();
    std::cout <<  "All done! \n";
    return 0;
}
