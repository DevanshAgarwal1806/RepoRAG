#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    // Constructor to initialize the object
    Person(std::string n, int a) : name(n), age(a) {}

    // Method to print the greeting
    void greet() {
        std::cout << "Hello, my name is " << name 
                  << " and I am " << age << " years old.\n";
    }
};

int main() {
    // Create an instance of Person using the constructor
    Person person1("Bob", 25);

    // Call the method on the object
    person1.greet();

    return 0;
}