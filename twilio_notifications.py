"""
Twilio WhatsApp Notification Module
Sends WhatsApp messages to suppliers for inventory management
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️  Twilio not available. Install with: pip install twilio")


def send_whatsapp_notification(to_number, message):
    """
    Send WhatsApp notification using Twilio

    Args:
        to_number: Phone number with country code (e.g., +1234567890)
        message: Message body to send

    Returns:
        dict with success status and message SID or error
    """
    if not TWILIO_AVAILABLE:
        return {
            "success": False,
            "error": "Twilio library not installed"
        }

    try:
        # Get Twilio credentials from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("WHATSAPP_FROM")

        if not all([account_sid, auth_token, from_number]):
            return {
                "success": False,
                "error": "Missing Twilio credentials. Check environment variables."
            }

        # Initialize Twilio client
        client = Client(account_sid, auth_token)

        # Format numbers for WhatsApp
        from_whatsapp = f"whatsapp:{from_number}"
        to_whatsapp = f"whatsapp:{to_number}"
        print(f"FROM: {from_whatsapp}")
        print(f"TO: {to_whatsapp}")
        print(f"📤 Sending WhatsApp notification...")
        # Send message
        message_obj = client.messages.create(
            body=message,
            from_=from_whatsapp,
            to=to_whatsapp
        )

        print(f"✅ Notification sent! SID: {message_obj.sid}")

        return {
            "success": True,
            "message_sid": message_obj.sid,
            "status": message_obj.status,
            "to": to_number
        }

    except Exception as e:
        print(f"❌ Error sending WhatsApp notification: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def format_supplier_notification(product_name, category, predicted_demand, optimal_qty):
    """
    Format a professional supplier notification message

    Args:
        product_name: Name of the product
        category: Product category
        predicted_demand: Predicted demand quantity
        optimal_qty: Optimal order quantity

    Returns:
        Formatted message string
    """
    message = f"""
🔔 *Inventory Alert*

📦 *Product:* {product_name}
📂 *Category:* {category}

📊 *ML-Powered Forecast:*
• Predicted Demand: {predicted_demand:.0f} units
• Recommended Order: {optimal_qty} units

Please confirm availability and delivery timeline.

---
_This is an automated message from your Inventory Management System._
    """.strip()

    return message


def format_bulk_notification(category, products_list):
    """
    Format a bulk notification for category-wide alerts with detailed product list.
    Splits messages if they exceed 1600 characters to comply with WhatsApp limits.

    Args:
        category: Product category
        products_list: List of dicts with product details [{'product_name': str, 'predicted_demand': float, 'optimal_qty': int}, ...]

    Returns:
        List of formatted message strings (split if necessary)
    """
    MAX_LENGTH = 1600
    total_products = len(products_list)
    total_predicted_demand = sum(p['predicted_demand'] for p in products_list)
    total_recommended_qty = sum(p['optimal_qty'] for p in products_list)

    # Header template
    header = f"""🔔 *Bulk Inventory Alert*

📂 *Category:* {category}
📦 *Products Affected:* {total_products}

📊 *Category Totals:*
• Total Predicted Demand: {total_predicted_demand:.0f} units
• Total Recommended Order: {total_recommended_qty} units"""

    footer = """
Please confirm availability and delivery timeline for the above items.

---
_This is an automated message from your Inventory Management System._"""

    # Calculate available space for products
    base_message_length = len(header) + len(footer) + len("\n\n📋 *Product Breakdown:*\n\n")

    # Build messages by grouping products
    messages = []
    current_products = []
    current_length = base_message_length

    for i, product in enumerate(products_list, 1):
        product_text = f"{i}. *{product['product_name']}*\n   • Predicted Demand: {product['predicted_demand']:.0f} units\n   • Recommended Order: {product['optimal_qty']} units"
        product_text_length = len(product_text) + 2  # +2 for newline separators

        # Check if adding this product would exceed the limit
        if current_length + product_text_length > MAX_LENGTH and current_products:
            # Create a message with current products
            products_section = "\n\n".join(current_products)
            message = f"""{header}

📋 *Product Breakdown (Part {len(messages) + 1}):*

{products_section}{footer}""".strip()
            messages.append(message)

            # Reset for next message
            current_products = [product_text]
            current_length = base_message_length + product_text_length
        else:
            current_products.append(product_text)
            current_length += product_text_length

    # Add remaining products
    if current_products:
        products_section = "\n\n".join(current_products)
        part_indicator = f" (Part {len(messages) + 1})" if messages else ""
        message = f"""{header}

📋 *Product Breakdown{part_indicator}:*

{products_section}{footer}""".strip()
        messages.append(message)

    return messages


def format_retailer_notification(product_name, status, requested_qty, supplier_name=None):
    """
    Format a notification message to retailer about supplier's decision

    Args:
        product_name: Name of the product
        status: Order status ("accepted" or "rejected")
        requested_qty: Quantity requested
        supplier_name: Optional supplier name

    Returns:
        Formatted message string
    """
    status_emoji = "✅" if status == "accepted" else "❌"
    status_text = "ACCEPTED" if status == "accepted" else "REJECTED"

    supplier_info = f" by {supplier_name}" if supplier_name else ""

    message = f"""
{status_emoji} *Order Update*

Your order has been *{status_text}*{supplier_info}

📦 *Product:* {product_name}
📊 *Quantity:* {requested_qty} units
📅 *Status:* {status_text}

{
'Thank you! Please coordinate delivery details with your supplier.' if status == 'accepted'
else 'Please contact your supplier for alternative arrangements.'
}

---
_This is an automated message from your Inventory Management System._
    """.strip()

    return message
