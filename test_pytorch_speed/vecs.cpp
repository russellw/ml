// results on a 3 GHz CPU
// counting only FP multiplies/sec

// vector length 10
// 10e6

// vector length 100
// 77e6

// vector length 100
// 714e6

// Suppress Microsoft C++ warnings in Torch header files
#ifdef _MSC_VER
#pragma warning(disable : 4067)
#pragma warning(disable : 4530)
#pragma warning(disable : 4624)
#pragma warning(disable : 4805)
#endif

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

int parse(char *s) {
  errno = 0;
  auto n = strtoull(s, 0, 10);
  if (errno) {
    std::cerr << strerror(errno) << '\n';
    exit(1);
  }
  return n;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: vecs <size> <times>\n";
    return 1;
  }
  auto n = parse(argv[1]);
  auto i = parse(argv[2]);
  auto a = torch::randn({n});
  auto b = torch::randn({n});
  at::Tensor c;
  	while(i--) {
     c=a.dot(b);
  }
  std::cout << c << '\n';
  return 0;
}
