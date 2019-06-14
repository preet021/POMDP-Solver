witness: main.o trim.o dot_product.o
	g++ -o witness $^

clean:
	@rm -f *.o witness

main.o: witness.h main.cpp
	$(CXX) -g -c main.cpp

trim.o: witness.h trim.cpp
	$(CXX) -g -c trim.cpp

dot_product.o: witness.h dot_product.cpp
	$(CXX) -g -c dot_product.cpp
