// Main.java
public class Main {
    public static void main(String[] args) {
        CustomerRepository repository = new CustomerRepository();
        CustomerService service = new CustomerService(repository);

        // Add some customers
        service.addCustomer(new Customer(1, "John Doe", "john@example.com"));
        service.addCustomer(new Customer(2, "Jane Doe", "jane@example.com"));

        // Retrieve a customer
        Customer customer = service.getCustomerById(1);
        System.out.println("Retrieved: " + customer);

        // Delete a customer
        service.deleteCustomer(2);
        System.out.println("Customer with ID 2 deleted.");
    }
}
