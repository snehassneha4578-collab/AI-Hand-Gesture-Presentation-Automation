import java.util.Optional;

public class Main {
    public static void main(String[] args) {
        Optional<String> name = Optional.of("Sneha");
        Optional<String> empty = Optional.empty();

        System.out.println(name);
        System.out.println(empty);
    }
}
