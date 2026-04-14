// utils.ts

export function formatDate(dateString: string): string {
    /** Formats a raw ISO date string into a readable local string. */
    const date = new Date(dateString);
    // Method call on standard JS object
    return date.toLocaleDateString();
}

export const calculateAge = (birthYear: number): number => {
    // Standard arrow function without a docstring
    const currentYear = new Date().getFullYear();
    return currentYear - birthYear;
}