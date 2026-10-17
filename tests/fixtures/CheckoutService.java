public class CheckoutService {
    public int calculateTotal(int price, int quantity, boolean discount) {
        int total = price * quantity;
        if (discount) {
            total = total - 10;
        }
        if (total < 0) {
            return 0;
        }
        return total;
    }
}
