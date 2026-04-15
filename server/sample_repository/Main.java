// File must be named Main.java to match the public class
public class Main {
    
    // Define the Person class as a static nested class for simplicity here,
    // though usually it would be in its own file (Person.java).
    static class Person {
        private String name;
        private int age;

        // Constructor
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }

        // Method to print the greeting
        public void greet() {
            System.out.println("Hello, my name is " + name + 
                               " and I am " + age + " years old.");
        }
    }

    // The main method is the entry point of the program
    public static void main(String[] args) {
        // Create a new Person object
        Person person1 = new Person("Charlie", 28);
        
        // Call the method
        person1.greet();
    }
}