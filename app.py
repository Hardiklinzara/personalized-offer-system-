import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Smart E-Commerce", layout="wide")

# ─────────────────────────────────────────
# LOAD ML MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("personalized_offer_model.pkl")

model = load_model()

# Label encoding maps (same order scikit-learn sees alphabetically)
ENCODE_MAP = {
    "gender":           {"Female": 0, "Male": 1},
    "location":         {"Bangalore": 0, "Chennai": 1, "Delhi": 2, "Hyderabad": 3, "Kolkata": 4, "Mumbai": 5},
    "category":         {"Dress": 0, "Hoodie": 1, "Jacket": 2, "Jeans": 3, "Shirt": 4, "Shoes": 5, "T-shirt": 6},
    "preferred_brand":  {"Adidas": 0, "H&M": 1, "Levis": 2, "Nike": 3, "Uniqlo": 4, "Zara": 5},
    "color_preference": {"Beige": 0, "Black": 1, "Blue": 2, "Green": 3, "Red": 4, "White": 5},
    "size":             {"L": 0, "M": 1, "S": 2, "XL": 3},
    "season":           {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3},
    "festival":         {"Christmas": 0, "Diwali": 1, "Eid": 2, "Holi": 3, "NewYear": 4},
    "discount_used":    {"No": 0, "Yes": 1},
    "price_sensitivity":{"High": 0, "Low": 1, "Medium": 2},
}

def predict_discount(profile: dict, product_price: float) -> tuple[int, float]:
    """
    Use the RandomForest model to predict offer probability.
    Returns (discount_pct, offer_probability).
    """
    row = [
        profile["age"],
        ENCODE_MAP["gender"][profile["gender"]],
        ENCODE_MAP["location"][profile["location"]],
        ENCODE_MAP["category"][profile["category"]],
        ENCODE_MAP["preferred_brand"][profile["preferred_brand"]],
        ENCODE_MAP["color_preference"][profile["color_preference"]],
        ENCODE_MAP["size"][profile["size"]],
        ENCODE_MAP["season"][profile["season"]],
        ENCODE_MAP["festival"][profile["festival"]],
        product_price,                           # purchase_amount
        profile["purchase_frequency"],
        profile["avg_order_value"],
        profile["last_purchase_days"],
        profile["browsing_time"],
        profile["product_views"],
        profile["wishlist_items"],
        ENCODE_MAP["discount_used"][profile["discount_used"]],
        profile["coupon_usage_rate"],
        ENCODE_MAP["price_sensitivity"][profile["price_sensitivity"]],
        profile["loyalty_score"],
    ]
    X = np.array(row).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]   # probability of getting an offer

    # Map probability → discount tier
    if proba >= 0.90:
        discount = 25
    elif proba >= 0.75:
        discount = 20
    elif proba >= 0.55:
        discount = 15
    elif proba >= 0.35:
        discount = 10
    else:
        discount = 5

    return discount, round(proba * 100, 1)


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state["role"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "products" not in st.session_state:
    st.session_state["products"] = [
        {"id": 1, "name": "Running Shoes", "price": 1999,
         "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
         "category": "Shoes"},
        {"id": 2, "name": "T-Shirt",       "price": 799,
         "image": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab",
         "category": "T-shirt"},
        {"id": 3, "name": "Jeans",         "price": 1499,
         "image": "https://images.unsplash.com/photo-1542272604-787c3835535d",
         "category": "Jeans"},
    ]

if "cart" not in st.session_state:
    st.session_state["cart"] = {}


# ─────────────────────────────────────────
# LOGIN  (now collects profile for ML)
# ─────────────────────────────────────────
def login():
    st.title("🛒 Smart AI E-Commerce")
    st.subheader("Login")

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Your Name")
        role     = st.selectbox("Role", ["Customer", "Shop Owner"])

    if role == "Customer":
        st.markdown("#### 👤 Your Shopping Profile *(helps AI personalise offers)*")
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.number_input("Age", 18, 80, 28)
            gender   = st.selectbox("Gender", ["Female", "Male"])
            location = st.selectbox("City", ["Bangalore","Chennai","Delhi",
                                             "Hyderabad","Kolkata","Mumbai"])
            season   = st.selectbox("Current Season", ["Summer","Winter","Autumn","Spring"])
            festival = st.selectbox("Upcoming Festival", ["Diwali","Christmas","Eid",
                                                          "Holi","NewYear"])
        with c2:
            preferred_brand  = st.selectbox("Favourite Brand", ["Nike","Adidas","Zara",
                                                                 "H&M","Levis","Uniqlo"])
            color_preference = st.selectbox("Colour Preference", ["Black","Blue","White",
                                                                   "Red","Green","Beige"])
            size             = st.selectbox("Size", ["S","M","L","XL"])
            price_sensitivity= st.selectbox("Price Sensitivity", ["Low","Medium","High"])
            discount_used    = st.selectbox("Used Discounts Before?", ["Yes","No"])
        with c3:
            loyalty_score      = st.slider("Loyalty Score (0–100)", 0.0, 100.0, 60.0)
            purchase_frequency = st.slider("Purchases / Month", 1, 30, 5)
            avg_order_value    = st.number_input("Avg Order Value (₹)", 200, 10000, 1200)
            last_purchase_days = st.number_input("Days Since Last Purchase", 1, 365, 15)
            browsing_time      = st.slider("Daily Browsing (mins)", 5, 120, 30)
            product_views      = st.slider("Products Viewed / Week", 1, 50, 10)
            wishlist_items     = st.slider("Wishlist Items", 0, 20, 4)
            coupon_usage_rate  = st.slider("Coupon Usage Rate", 0.0, 1.0, 0.5)

    if st.button("Login"):
        if not username:
            st.warning("Please enter your name.")
            return
        st.session_state["role"]     = role
        st.session_state["username"] = username
        if role == "Customer":
            st.session_state["user_profile"] = {
                "age": age, "gender": gender, "location": location,
                "preferred_brand": preferred_brand, "color_preference": color_preference,
                "size": size, "season": season, "festival": festival,
                "purchase_frequency": purchase_frequency, "avg_order_value": avg_order_value,
                "last_purchase_days": last_purchase_days, "browsing_time": browsing_time,
                "product_views": product_views, "wishlist_items": wishlist_items,
                "discount_used": discount_used, "coupon_usage_rate": coupon_usage_rate,
                "price_sensitivity": price_sensitivity, "loyalty_score": loyalty_score,
                "category": "T-shirt",   # default; overridden per product
            }
        st.rerun()


# ─────────────────────────────────────────
# CUSTOMER DASHBOARD
# ─────────────────────────────────────────
def customer_dashboard():
    st.title(f"Welcome {st.session_state['username']} 👋")

    profile = st.session_state["user_profile"]

    # Sidebar cart summary
    st.sidebar.header("🛒 Cart")
    item_count = sum(st.session_state["cart"].values())
    st.sidebar.write(f"Items in cart: **{item_count}**")

    search = st.text_input("🔍 Search Products")
    st.markdown("## 🤖 AI-Personalised Offers For You")

    cols = st.columns(3)

    for i, product in enumerate(st.session_state["products"]):
        if search and search.lower() not in product["name"].lower():
            continue

        # Override profile category to match product
        product_profile = {**profile, "category": product.get("category", "T-shirt")}

        discount, ai_score = predict_discount(product_profile, product["price"])

        final_price = int(product["price"] * (1 - discount / 100))
        saved       = product["price"] - final_price

        with cols[i % 3]:
            st.image(product["image"], use_container_width=True)
            st.markdown(f"### {product['name']}")
            st.markdown(
                f"~~₹{product['price']}~~  \n"
                f"**₹{final_price} ({discount}% OFF)**  \n"
                f"💸 You save ₹{saved}"
            )
            st.caption(f"🤖 AI Offer Score: {ai_score}%")

            if discount >= 20:
                st.caption("🔥 Top deal for you!")
            elif discount >= 15:
                st.caption("⭐ Great personalised offer")

            if st.button("Add to Cart", key=f"add{i}"):
                pid = product["id"]
                st.session_state["cart"][pid] = st.session_state["cart"].get(pid, 0) + 1
                st.rerun()

    st.divider()
    st.markdown("## 🧾 Cart Details")

    if not st.session_state["cart"]:
        st.info("Your cart is empty.")
        return

    total = 0
    total_savings = 0

    for pid, qty in list(st.session_state["cart"].items()):
        product = next(p for p in st.session_state["products"] if p["id"] == pid)
        product_profile = {**profile, "category": product.get("category", "T-shirt")}
        discount, _ = predict_discount(product_profile, product["price"])
        final_price = int(product["price"] * (1 - discount / 100))

        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.write(f"{product['name']} (×{qty}) — ₹{final_price * qty}")
        with col2:
            if st.button("➕", key=f"inc{pid}"):
                st.session_state["cart"][pid] += 1
                st.rerun()
        with col3:
            if st.button("❌", key=f"dec{pid}"):
                st.session_state["cart"][pid] -= 1
                if st.session_state["cart"][pid] <= 0:
                    del st.session_state["cart"][pid]
                st.rerun()

        total         += final_price * qty
        total_savings += (product["price"] - final_price) * qty

    st.subheader(f"Total: ₹{total}")
    st.success(f"🎉 You saved ₹{total_savings} with AI personalisation!")

    if st.button("Clear Cart"):
        st.session_state["cart"] = {}
        st.rerun()


# ─────────────────────────────────────────
# OWNER DASHBOARD  (unchanged)
# ─────────────────────────────────────────
def owner_dashboard():
    st.title(f"Shop Owner Panel 👨‍💼 ({st.session_state['username']})")

    st.header("➕ Add Product")
    name     = st.text_input("Product Name")
    price    = st.number_input("Price", min_value=0)
    image    = st.text_input("Image URL")
    category = st.selectbox("Category", ["Dress","Hoodie","Jacket","Jeans",
                                         "Shirt","Shoes","T-shirt"])

    if st.button("Add Product"):
        if name and price and image:
            st.session_state["products"].append({
                "id": len(st.session_state["products"]) + 1,
                "name": name, "price": price,
                "image": image, "category": category,
            })
            st.success("Product Added ✅")
            st.rerun()
        else:
            st.warning("Please fill all fields.")

    st.divider()
    st.header("📦 Your Products")
    cols = st.columns(3)
    for i, product in enumerate(st.session_state["products"]):
        with cols[i % 3]:
            st.image(product["image"])
            st.write(f"{product['name']} — ₹{product['price']}")
            st.caption(f"Category: {product.get('category','—')}")
            if st.button("Delete", key=f"del{i}"):
                st.session_state["products"].pop(i)
                st.rerun()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    if st.session_state["role"] is None:
        login()
    else:
        st.sidebar.title("Navigation")
        if st.sidebar.button("Logout"):
            st.session_state["role"]    = None
            st.session_state["cart"]    = {}
            st.session_state["user_profile"] = None
            st.rerun()

        if st.session_state["role"] == "Customer":
            customer_dashboard()
        else:
            owner_dashboard()

main()
