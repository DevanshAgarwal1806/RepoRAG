/**
 * Fetches legacy user profile data from the v1 REST API.
 * Needs to be deprecated by Q3.
 */
export async function fetchLegacyUserData(userId) {
    try {
        const response = await fetch(`/api/v1/users/${userId}`);
        return await response.json();
    } catch (error) {
        console.error("Failed to fetch user:", error);
        return null;
    }
}
