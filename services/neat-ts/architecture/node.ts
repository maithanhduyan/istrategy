/**
 * Node class for NEAT-TS
 * @module Node
 */
import config from '../config';
import methods from '../methods/methods';
import Connection from './connection';

/**
 * Represents a node in a neural network.
 */
class Node {
    bias: number;
    squash: any;
    type: string;
    activation: number = 0;
    state: number = 0;
    old: number = 0;
    mask: number = 1;
    previousDeltaBias: number = 0;
    totalDeltaBias: number = 0;
    connections: {
        in: any[];
        out: any[];
        gated: any[];
        self: any;
    };
    error: {
        responsibility: number;
        projected: number;
        gated: number;
    };
    derivative?: number;

    /**
     * Index of the node in the network (used internally for serialization).
     */
    index?: number;

    constructor(type?: string) {
        this.bias = (type === 'input') ? 0 : Math.random() * 0.2 - 0.1;
        this.squash = methods.activation.LOGISTIC;
        this.type = type || 'hidden';
        this.activation = 0;
        this.state = 0;
        this.old = 0;
        this.mask = 1;
        this.previousDeltaBias = 0;
        this.totalDeltaBias = 0;
        this.connections = {
            in: [],
            out: [],
            gated: [],
            self: new Connection(this, this, 0)
        };
        this.error = {
            responsibility: 0,
            projected: 0,
            gated: 0
        };
    }

    /**
     * Activates the node
     */
    activate(input?: number): number {
        // ...existing code...
        if (typeof input !== 'undefined') {
            this.activation = input;
            return this.activation;
        }
        this.old = this.state;
        this.state = this.connections.self.gain * this.connections.self.weight * this.state + this.bias;
        for (let i = 0; i < this.connections.in.length; i++) {
            let connection = this.connections.in[i];
            this.state += connection.from.activation * connection.weight * connection.gain;
        }
        this.activation = this.squash(this.state) * this.mask;
        this.derivative = this.squash(this.state, true);
        // ...existing code for traces...
        let nodes: any[] = [];
        let influences: any[] = [];
        for (let i = 0; i < this.connections.gated.length; i++) {
            let conn = this.connections.gated[i];
            let node = conn.to;
            let index = nodes.indexOf(node);
            if (index > -1) {
                influences[index] += conn.weight * conn.from.activation;
            } else {
                nodes.push(node);
                influences.push(conn.weight * conn.from.activation + (node.connections.self.gater === this ? node.old : 0));
            }
            conn.gain = this.activation;
        }
        for (let i = 0; i < this.connections.in.length; i++) {
            let connection = this.connections.in[i];
            connection.elegibility = this.connections.self.gain * this.connections.self.weight * connection.elegibility + connection.from.activation * connection.gain;
            for (let j = 0; j < nodes.length; j++) {
                let node = nodes[j];
                let influence = influences[j];
                let index = connection.xtrace.nodes.indexOf(node);
                if (index > -1) {
                    if (node.connections && node.connections.self && connection.xtrace && connection.xtrace.values) {
                        connection.xtrace.values[index] = node.connections.self.gain * node.connections.self.weight * connection.xtrace.values[index] + this.derivative! * connection.elegibility * influence;
                    }
                } else {
                    if (connection.xtrace && connection.xtrace.values) {
                        connection.xtrace.nodes.push(node);
                        connection.xtrace.values.push(this.derivative! * connection.elegibility * influence);
                    }
                }
            }
        }
        return this.activation;
    }

    /**
     * Activates the node without calculating elegibility traces and such
     */
    noTraceActivate(input?: number): number {
        if (typeof input !== 'undefined') {
            this.activation = input;
            return this.activation;
        }
        this.state = this.connections.self.gain * this.connections.self.weight * this.state + this.bias;
        for (let i = 0; i < this.connections.in.length; i++) {
            let connection = this.connections.in[i];
            this.state += connection.from.activation * connection.weight * connection.gain;
        }
        this.activation = this.squash(this.state);
        for (let i = 0; i < this.connections.gated.length; i++) {
            this.connections.gated[i].gain = this.activation;
        }
        return this.activation;
    }

    /**
     * Back-propagate the error, aka learn
     */
    propagate(rate?: number, momentum?: number, update?: boolean, target?: number): void {
        momentum = momentum || 0;
        rate = rate || 0.3;
        let error = 0;
        if (this.type === 'output') {
            this.error.responsibility = this.error.projected = (target as number) - this.activation;
        } else {
            for (let i = 0; i < this.connections.out.length; i++) {
                let connection = this.connections.out[i];
                let node = connection.to;
                error += node.error.responsibility * connection.weight * connection.gain;
            }
            this.error.projected = this.derivative! * error;
            error = 0;
            for (let i = 0; i < this.connections.gated.length; i++) {
                let conn = this.connections.gated[i];
                let node = conn.to;
                let influence = node.connections.self.gater === this ? node.old : 0;
                influence += conn.weight * conn.from.activation;
                error += node.error.responsibility * influence;
            }
            this.error.gated = this.derivative! * error;
            this.error.responsibility = this.error.projected + this.error.gated;
        }
        if (this.type === 'constant') return;
        for (let i = 0; i < this.connections.in.length; i++) {
            let connection = this.connections.in[i];
            let gradient = this.error.projected * connection.elegibility;
            for (let j = 0; j < connection.xtrace.nodes.length; j++) {
                let node = connection.xtrace.nodes[j];
                let value = connection.xtrace.values[j];
                gradient += node.error.responsibility * value;
            }
            let deltaWeight = rate * gradient * this.mask;
            connection.totalDeltaWeight += deltaWeight;
            if (update) {
                connection.totalDeltaWeight += momentum * connection.previousDeltaWeight;
                connection.weight += connection.totalDeltaWeight;
                connection.previousDeltaWeight = connection.totalDeltaWeight;
                connection.totalDeltaWeight = 0;
            }
        }
        let deltaBias = rate * this.error.responsibility;
        this.totalDeltaBias += deltaBias;
        if (update) {
            this.totalDeltaBias += momentum * this.previousDeltaBias;
            this.bias += this.totalDeltaBias;
            this.previousDeltaBias = this.totalDeltaBias;
            this.totalDeltaBias = 0;
        }
    }

    /**
     * Creates a connection from this node to the given node
     */
    connect(target: any, weight?: number): any[] {
        let connections: any[] = [];
        if (typeof target.bias !== 'undefined') {
            if (target === this) {
                if (this.connections.self.weight !== 0) {
                    if (config.warnings) console.warn('This connection already exists!');
                } else {
                    this.connections.self.weight = weight || 1;
                }
                connections.push(this.connections.self);
            } else if (this.isProjectingTo(target)) {
                throw new Error('Already projecting a connection to this node!');
            } else {
                let connection = new Connection(this, target, weight);
                target.connections.in.push(connection);
                this.connections.out.push(connection);
                connections.push(connection);
            }
        } else {
            for (let i = 0; i < target.nodes.length; i++) {
                let connection = new Connection(this, target.nodes[i], weight);
                target.nodes[i].connections.in.push(connection);
                this.connections.out.push(connection);
                target.connections.in.push(connection);
                connections.push(connection);
            }
        }
        return connections;
    }

    /**
     * Disconnects this node from the other node
     */
    disconnect(node: any, twosided?: boolean): void {
        if (this === node) {
            this.connections.self.weight = 0;
            return;
        }
        for (let i = 0; i < this.connections.out.length; i++) {
            let conn = this.connections.out[i];
            if (conn.to === node) {
                this.connections.out.splice(i, 1);
                let j = conn.to.connections.in.indexOf(conn);
                conn.to.connections.in.splice(j, 1);
                if (conn.gater !== null) conn.gater.ungate(conn);
                break;
            }
        }
        if (twosided) {
            node.disconnect(this);
        }
    }

    /**
     * Make this node gate a connection
     */
    gate(connections: any[] | any): void {
        if (!Array.isArray(connections)) {
            connections = [connections];
        }
        for (let i = 0; i < connections.length; i++) {
            let connection = connections[i];
            this.connections.gated.push(connection);
            connection.gater = this;
        }
    }

    /**
     * Removes the gates from this node from the given connection(s)
     */
    ungate(connections: any[] | any): void {
        if (!Array.isArray(connections)) {
            connections = [connections];
        }
        for (let i = connections.length - 1; i >= 0; i--) {
            let connection = connections[i];
            let index = this.connections.gated.indexOf(connection);
            this.connections.gated.splice(index, 1);
            connection.gater = null;
            connection.gain = 1;
        }
    }

    /**
     * Clear the context of the node
     */
    clear(): void {
        for (let i = 0; i < this.connections.in.length; i++) {
            let connection = this.connections.in[i];
            connection.elegibility = 0;
            connection.xtrace = {
                nodes: [],
                values: []
            };
        }
        for (let i = 0; i < this.connections.gated.length; i++) {
            let conn = this.connections.gated[i];
            conn.gain = 0;
        }
        this.error.responsibility = this.error.projected = this.error.gated = 0;
        this.old = this.state = this.activation = 0;
    }

    /**
     * Mutates the node with the given method
     */
    mutate(method: any): void {
        if (typeof method === 'undefined') {
            throw new Error('No mutate method given!');
        } else if (!(method.name in methods.mutation)) {
            throw new Error('This method does not exist!');
        }
        switch (method) {
            case methods.mutation.MOD_ACTIVATION: {
                let squash = method.allowed[(method.allowed.indexOf(this.squash) + Math.floor(Math.random() * (method.allowed.length - 1)) + 1) % method.allowed.length];
                this.squash = squash;
                break;
            }
            case methods.mutation.MOD_BIAS: {
                let modification = Math.random() * (method.max - method.min) + method.min;
                this.bias += modification;
                break;
            }
        }
    }

    /**
     * Checks if this node is projecting to the given node
     */
    isProjectingTo(node: any): boolean {
        if (node === this && this.connections.self.weight !== 0) return true;
        for (let i = 0; i < this.connections.out.length; i++) {
            let conn = this.connections.out[i];
            if (conn.to === node) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks if the given node is projecting to this node
     */
    isProjectedBy(node: any): boolean {
        if (node === this && this.connections.self.weight !== 0) return true;
        for (let i = 0; i < this.connections.in.length; i++) {
            let conn = this.connections.in[i];
            if (conn.from === node) {
                return true;
            }
        }
        return false;
    }

    /**
     * Converts the node to a json object
     */
    toJSON(): any {
        let json = {
            bias: this.bias,
            type: this.type,
            squash: this.squash.name,
            mask: this.mask
        };
        return json;
    }

    /**
     * Convert a json object to a node
     */
    static fromJSON(json: any): Node {
        let node = new Node();
        node.bias = json.bias;
        node.type = json.type;
        node.mask = json.mask;
        node.squash = (methods.activation as any)[json.squash];
        return node;
    }
}

export default Node;