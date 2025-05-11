// CustomerRepository.java
import java.util.HashMap;
import java.util.Map;

public class CustomerRepository {
    private Map<Integer, Customer> customerMap = new HashMap<>();

    public void save(Customer customer) {
        customerMap.put(customer.getId(), customer);
    }

    public Customer findById(int id) {
        return customerMap.get(id);
    }

    public void deleteById(int id) {
        customerMap.remove(id);
    }
}
