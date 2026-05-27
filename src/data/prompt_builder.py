"""
Prompt building for LLM marketing copy generation.
"""

import pandas as pd


def build_full_prompt(customer: pd.Series, product: pd.Series) -> str:
    """
    Build a prompt for the LLM to generate personalized marketing copy.
    
    Args:
        customer: Series containing customer data
        product: Series containing product data
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a marketing copywriter. Generate a personalized marketing message (2-3 sentences) for this customer.

Customer Segment Details:
- Age: {customer['age']}
- Gender: {customer['gender']}
- Income (USD): {customer['income_usd']}
- Education: {customer['education']}
- Occupation: {customer['occupation']}
- Family Size: {customer['family_size']}
- Marital Status: {customer['marital_status']}
- Country: {customer['country']}
- City Type: {customer['city_type']}
- Climate Zone: {customer['climate_zone']}

- Lifestyle: {customer['lifestyle']}
- Personality: {customer['personality']}
- Primary Value: {customer['primary_value']}

- Purchase Frequency / Month: {customer['purchase_frequency_per_month']}
- Average Order Value (USD): {customer['avg_order_value_usd']}
- Total Orders: {customer['total_orders']}
- Total Spent (USD): {customer['total_spent_usd']}
- Days Since Last Purchase: {customer['last_purchase_days_ago']}
- Preferred Channel: {customer['preferred_channel']}
- Product Category Preference: {customer['product_category_pref']}
- Customer Lifetime Value (USD): {customer['customer_lifetime_value_usd']}
- Primary Motivation: {customer['primary_motivation']}

Product Details:
- Name: {product['product_name']}
- Category: {product['category']}
- Features: {product['core_attributes_/_features']}
- Tagline: {product['marketing_message']}

Instructions: 
Write a friendly, persuasive, and personalized marketing message for this customer. 
Do not repeat the raw data; focus on benefits, personalization, and engaging tone.
"""


def generate_marketing_copies(
    customers: pd.DataFrame, products: pd.DataFrame, llm_client, llm_model: str,
    temperature: float = 0.8, max_tokens: int = 120
) -> pd.DataFrame:
    """
    Generate marketing copies for all customers using the LLM.
    
    Args:
        customers: DataFrame with customer data
        products: DataFrame with product data
        llm_client: OpenAI client instance
        llm_model: Model name to use
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens for generation
    
    Returns:
        DataFrame with generated marketing copies
    """
    from tqdm import tqdm
    from src.data.loader import break_text_by_words

    copies = []
    for _idx, customer in tqdm(customers.iterrows(), total=len(customers)):
        products_sample = products.sample(1)
        for _p_idx, product in products_sample.iterrows():
            prompt = build_full_prompt(customer, product)

            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            message = response.choices[0].message.content.strip()

            copies.append(
                {
                    "segment": customer["segment"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "P": 0,
                    "H": 0,
                    "C": 0,
                    "G": 0,
                    "R": 0,
                    "prompt": break_text_by_words(prompt),
                    "marketing_copy": message,
                }
            )

    df_copy = pd.DataFrame(copies)
    return df_copy