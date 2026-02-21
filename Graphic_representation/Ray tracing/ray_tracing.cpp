# include <iostream>
# include <fstream>
# include <cmath>
# define PI 3.1415926535

int main() {
    int nx = 400;
    int ny = 400;
    std::ofstream file("colors.ppm");
    file << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i <nx; i++) {
            float r = float(i) / float(nx) * PI;
            float g = float(j) / float(ny) * PI;
            float b = 1.0;
            int ir = int(255.99 * sin(r));
            int ig = int(255.99 * sin(g));
            int ib = int(255.99 * sin(b));
            file << ir << " " << ig << " " << ib << "\n";
        }
    }
    file.close();
    return 0;
}