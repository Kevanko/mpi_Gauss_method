# Имя выходного файла
TARGET = gaussian

# Компилятор и флаги
CC = mpic++
CFLAGS = -O2 -std=c++20

# Исходный файл
SRC = gaussian.cpp

# Правила
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: all
	mpirun -np 4 ./$(TARGET)
