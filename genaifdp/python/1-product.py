class Product:
    def __init__(self, code, name, supplier, price):
        self.code = code
        self.name = name
        self.supplier = supplier
        self.price = price

    def info(self):
        print(f"Code: {self.code}")
        print(f"Name: {self.name}")
        print(f"Supplier: {self.supplier}")
        print(f"Price: {self.price}")
        print("____________________________________")

code = input("Enter product code: ")
name = input("Enter product name: ")
supplier = input("Enter supplier name: ")
price = float(input("Enter product price: "))

product1 = Product(code, name, supplier, price)
#product1.info()
code = input("Enter product code: ")
name = input("Enter product name: ")
supplier = input("Enter supplier name: ")
price = float(input("Enter product price: "))

product2 = Product(code, name, supplier, price)
#product2.info()
code = input("Enter product code: ")
name = input("Enter product name: ")
supplier = input("Enter supplier name: ")
price = float(input("Enter product price: "))

product3 = Product(code, name, supplier, price)
#product3.info()
class ProductManagement:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)

    def list_products(self):
        for product in self.products:
            product.info()
            print()  # Add a newline for better readability

    def find_product(self, name):
        for product in self.products:
            if product.name == name:
                print("Product found:")
            else:
                continue       
                print("Product not found.") 

    def findebypricerange(self, min_price, max_price):
        found = False
        for product in self.products:
            if min_price <= product.price <= max_price:
                product.info()
                found = True
        if not found:
            print("No products found in this price range.")
  
ProductManagement = ProductManagement()
ProductManagement.add_product(product1)
ProductManagement.add_product(product2)
ProductManagement.add_product(product3)         
ProductManagement.list_products()
print("_Find Product by Name_")
ProductManagement.find_product("Smartphone")  
print("_Find Product by price range_") 
ProductManagement.findebypricerange(500, 900)
ProductManagement.list_products()  # Uncomment to list products again