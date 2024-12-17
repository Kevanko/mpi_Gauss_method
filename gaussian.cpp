#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

// Обычная версия вычисления определителя методом Гаусса
double gaussianDeterminant(vector<vector<double>> matrix)
{
  int n = matrix.size();
  double det = 1.0;

  for (int i = 0; i < n; ++i)
  {
    int maxRow = i;
    for (int k = i + 1; k < n; ++k)
    {
      if (fabs(matrix[k][i]) > fabs(matrix[maxRow][i]))
      {
        maxRow = k;
      }
    }

    if (matrix[maxRow][i] == 0)
    {
      return 0.0;
    }

    if (i != maxRow)
    {
      swap(matrix[i], matrix[maxRow]);
      det *= -1;
    }

    det *= matrix[i][i];

    for (int k = i + 1; k < n; ++k)
    {
      double factor = matrix[k][i] / matrix[i][i];
      for (int j = i; j < n; ++j)
      {
        matrix[k][j] -= factor * matrix[i][j];
      }
    }
  }

  return det;
}

double parallelGaussianDeterminant(vector<vector<double>> matrix)
{
  int n = matrix.size();
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double det = 1.0;

  for (int i = 0; i < n; ++i)
  {
    int maxRow = i;
    double maxValue = 0.0;

    // Процесс 0 находит строку с максимальным элементом
    if (rank == 0)
    {
      for (int k = i; k < n; ++k)
      {
        if (fabs(matrix[k][i]) > maxValue)
        {
          maxRow = k;
          maxValue = fabs(matrix[k][i]);
        }
      }
    }

    // Рассылка номера строки с максимальным элементом
    MPI_Bcast(&maxRow, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Перестановка строк
    if (maxRow != i)
    {
      if (rank == 0)
      {
        swap(matrix[i], matrix[maxRow]);
      }
      det *= -1; // Меняем знак определителя
    }

    // Рассылка текущей строки i всем процессам
    MPI_Bcast(matrix[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (matrix[i][i] == 0)
    {
      return 0.0; // Определитель равен 0, если диагональный элемент равен 0
    }

    det *= matrix[i][i];

    // Обновление строк, распределённых между процессами
    for (int k = i + 1 + rank; k < n; k += size)
    {
      double factor = matrix[k][i] / matrix[i][i];
      for (int j = i; j < n; ++j)
      {
        matrix[k][j] -= factor * matrix[i][j];
      }
    }
  }

  // Объединение результатов определителя
  MPI_Allreduce(MPI_IN_PLACE, &det, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
  return det;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  vector<vector<double>> matrix = {
      {2, 1, 1},
      {1, 3, 2},
      {1, 0, 0}};

  double result = parallelGaussianDeterminant(matrix);

  if (rank == 0)
  {
    cout << "Определитель матрицы (параллельный): " << result << endl;

    result = gaussianDeterminant(matrix);
    cout << "Определитель матрицы (обычный): " << result << endl;
  }

  MPI_Finalize();
  return 0;
}
