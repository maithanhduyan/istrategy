// Converted from js/architecture/connection.js to TypeScript
/**
 * Connection class for NEAT-TS
 * @module Connection
 */


class Connection {
    from: any;
    to: any;
    gain: number = 1;
    weight: number;
    gater: any = null;
    elegibility: number = 0;
    previousDeltaWeight: number = 0;
    totalDeltaWeight: number = 0;
    xtrace: { nodes: any[]; values: any[] } = { nodes: [], values: [] };

    constructor(from: any, to: any, weight?: number) {
        this.from = from;
        this.to = to;
        this.gain = 1;
        this.weight = (typeof weight === 'undefined') ? Math.random() * 0.2 - 0.1 : weight;
        this.gater = null;
        this.elegibility = 0;
        this.previousDeltaWeight = 0;
        this.totalDeltaWeight = 0;
        this.xtrace = { nodes: [], values: [] };
    }

    /**
     * Converts the connection to a json object
     */
    toJSON(): any {
        let json = {
            weight: this.weight
        };
        return json;
    }

    /**
     * Returns an innovation ID (Cantor pairing function)
     * @see https://en.wikipedia.org/wiki/Pairing_function
     */
    static innovationID(a: number, b: number): number {
        return 0.5 * (a + b) * (a + b + 1) + b;
    }
}

export default Connection;
