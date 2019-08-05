CC = g++ -std=c++14

witness: main.o trim.o dot_product.o get_zmdp_policy.o
	$(CC) -o witness $^

clean:
	@rm -f *.o witness *.lp *.out

main.o: witness.h main.cpp
	$(CC) -g -c main.cpp

trim.o: witness.h trim.cpp
	$(CC) -g -c trim.cpp

dot_product.o: witness.h dot_product.cpp
	$(CC) -g -c dot_product.cpp

get_zmdp_policy.o: witness.h get_zmdp_policy.cpp
	$(CC) -g -c get_zmdp_policy.cpp
