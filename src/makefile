CC = g++ -std=c++14

witness: main.o trim.o dot_product.o
	$(CC) -o witness $^

clean:
	@rm -f *.o witness *.lp

main.o: witness.h main.cpp
	$(CC) -g -c main.cpp

trim.o: witness.h trim.cpp
	$(CC) -g -c trim.cpp

dot_product.o: witness.h dot_product.cpp
	$(CC) -g -c dot_product.cpp
