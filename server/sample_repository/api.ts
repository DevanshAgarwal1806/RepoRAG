// api.ts

export class ApiClient {
    baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async fetchUserData(id: string) {
        /** Fetches user data from the backend API. */
        // Global fetch call
        const response = await fetch(`${this.baseUrl}/users/${id}`);
        // Object method call
        return response.json();
    }
}