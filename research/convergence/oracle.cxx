// Reference ITK monitor; not part of the voxreg runtime or build.
#include "itkWindowConvergenceMonitoringFunction.h"
#include <iomanip>
#include <iostream>

int main(int argc, char **argv)
{
  if (argc != 2 || std::stoul(argv[1]) < 2)
    return 2;
  auto monitor = itk::Function::WindowConvergenceMonitoringFunction<double>::New();
  monitor->SetWindowSize(std::stoul(argv[1]));
  double loss;
  std::cout << std::setprecision(17);
  while (std::cin >> loss)
  {
    monitor->AddEnergyValue(loss);
    std::cout << monitor->GetConvergenceValue() << '\n';
  }
}
