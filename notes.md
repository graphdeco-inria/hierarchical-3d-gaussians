# Notes on Handling Multiple Strategies for Top HCP Identification

## Approach to Handle Multiple Strategies

To modify the `/api/healthcare-professionals/top` route to handle multiple strategies, the following approach can be taken:

1. **Accept an Array of Strategies**:
   - Modify the API to accept an array of strategies as a query parameter. Example: `/api/healthcare-professionals/top?limit=3&strategies=elephant,potential`.

2. **Implement Strategy Calculation**:
   - Define how each strategy calculates the relevance of an HCP. For instance:
     - **Elephant Strategy**: Focuses on the gap between procedures performed and device sales.
     - **Potential Strategy**: Focuses on HCPs who have a high number of procedures but low current usage of the device, indicating potential for increased sales.

3. **Aggregate Scores from Multiple Strategies**:
   - For each HCP, compute a combined score based on the chosen strategies. One approach could be to assign weights to each strategy and calculate a weighted sum of the scores.

4. **Sorting HCPs**:
   - Sort HCPs based on their aggregated scores. If scores are tied, sort by HCP ID as a secondary criterion.

## Example Query Parameters

- `limit`: Number of top HCPs to return.
- `strategies`: Array of strategies to consider (e.g., `["elephant", "potential"]`).
- `weights` (optional): Weights assigned to each strategy (e.g., `{ "elephant": 0.5, "potential": 0.5 }`).

By implementing these changes, the service can provide more flexible and comprehensive identification of top HCPs based on various strategic considerations.
