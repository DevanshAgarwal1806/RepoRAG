#include <stdio.h>
#include <string.h>

// Define a struct to hold person data
struct Person {
    char name[50];
    int age;
};

// Function to print the greeting
void greet(struct Person p) {
    printf("Hello, my name is %s and I am %d years old.\n", p.name, p.age);
}

int main() {
    // Create and initialize a Person instance
    struct Person person1;
    strcpy(person1.name, "Alice");
    person1.age = 30;

    // Call the greeting function
    greet(person1);

    return 0;
}