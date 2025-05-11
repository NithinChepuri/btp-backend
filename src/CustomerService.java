// CustomerService.java
public class CustomerService {
    private CustomerRepository repository;

    public CustomerService(CustomerRepository repository) {
        this.repository = repository;
    }

    public Customer getCustomerById(int id) {
        return repository.findById(id);
    }

    public void addCustomer(Customer customer) {
        repository.save(customer);
    }

    public void deleteCustomer(int id) {
        repository.deleteById(id);
    }
}
