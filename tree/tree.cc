#include <string>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

int main(int argc,char**argv) {
    std::string path = argv[1];
    for (const auto & entry : fs::recursive_directory_iterator(path))
        std::cout << entry.path() << std::endl;
}
