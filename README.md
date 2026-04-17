Task 1: I replaced the hardcoded pricing step with a dynamic tool-calling approach. The request text is parsed to extract the quantity, and the LLM is bound to a get_unit_price(vendor) tool. The model calls the tool once per vendor, and the node calculates total = unit_price * quantity for each quote.

Task 2: I added conditional routing after compare_quotes. If the best quote exceeds €10,000, the graph routes to request_approval. Otherwise, it skips approval and proceeds directly to purchase order submission.

Task 3: I changed the rejection handling so that a rejected approval becomes a proper graph outcome. If the manager rejects the purchase, the graph skips submit_purchase_order and goes directly to notify_employee, passing the rejection reason in state.

Task 4: I replaced hardcoded unit prices with live data from DummyJSON. The tool fetches laptop products from the API, searches for the cheapest suitable product available within 2 weeks, and uses that product’s price and details in the quote and approval flow. If no suitable match is found, it falls back to a default price and logs a warning.
