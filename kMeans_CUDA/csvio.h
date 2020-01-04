#include <vector>
#include <string>

void write2VecTo(std::string filename, std::string delimiter, std::vector<float>& vec);
void read2VecFrom(std::string filename, std::string delimiter, std::vector<float>& dest);
void initializeClusters(int dimension, std::vector<float>& clusters_x, std::vector<float>& clusters_y, std::vector<float>& points);