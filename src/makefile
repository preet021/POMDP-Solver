witness: main.o
	g++ -o witness $^

clean:
	@rm -f *.o witness

main.o: witness.h main.cpp
	$(CXX) -g -c main.cpp
