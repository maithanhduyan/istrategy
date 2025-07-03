// Converted from js/architecture/network.js to TypeScript
import config from '../config';
import methods from '../methods/methods';
import Node from './node';

const mutation = methods.mutation;

class Network {
    input: number;
    output: number;
    nodes: any[] = [];
    connections: any[] = [];
    gates: any[] = [];
    selfconns: any[] = [];
    dropout: number = 0;

    constructor(input: number, output: number) {
        if (typeof input === 'undefined' || typeof output === 'undefined') {
            throw new Error('No input or output size given');
        }
        this.input = input;
        this.output = output;
        this.nodes = [];
        this.connections = [];
        this.gates = [];
        this.selfconns = [];
        this.dropout = 0;
        let i: number;
        for (i = 0; i < this.input + this.output; i++) {
            let type = i < this.input ? 'input' : 'output';
            this.nodes.push(new Node(type));
        }
        for (let i = 0; i < this.input; i++) {
            for (let j = this.input; j < this.output + this.input; j++) {
                let weight = Math.random() * this.input * Math.sqrt(2 / this.input);
                this.connect(this.nodes[i], this.nodes[j], weight);
            }
        }
    }

    activate(input: number[], training?: boolean): number[] {
        let output: number[] = [];
        for (let i = 0; i < this.nodes.length; i++) {
            if (this.nodes[i].type === 'input') {
                this.nodes[i].activate(input[i]);
            } else if (this.nodes[i].type === 'output') {
                let activation = this.nodes[i].activate();
                output.push(activation);
            } else {
                if (training) this.nodes[i].mask = Math.random() < this.dropout ? 0 : 1;
                this.nodes[i].activate();
            }
        }
        return output;
    }

    noTraceActivate(input: number[]): number[] {
        let output: number[] = [];
        for (let i = 0; i < this.nodes.length; i++) {
            if (this.nodes[i].type === 'input') {
                this.nodes[i].noTraceActivate(input[i]);
            } else if (this.nodes[i].type === 'output') {
                let activation = this.nodes[i].noTraceActivate();
                output.push(activation);
            } else {
                this.nodes[i].noTraceActivate();
            }
        }
        return output;
    }

    propagate(rate?: number, momentum?: number, update?: boolean, target?: number[]): void {
        if (typeof target === 'undefined' || target.length !== this.output) {
            throw new Error('Output target length should match network output length');
        }
        let targetIndex = target.length;
        for (let i = this.nodes.length - 1; i >= this.nodes.length - this.output; i--) {
            this.nodes[i].propagate(rate, momentum, update, target[--targetIndex]);
        }
        for (let i = this.nodes.length - this.output - 1; i >= this.input; i--) {
            this.nodes[i].propagate(rate, momentum, update);
        }
    }

    clear(): void {
        for (let i = 0; i < this.nodes.length; i++) {
            this.nodes[i].clear();
        }
    }

    connect(from: any, to: any, weight?: number): any[] {
        let connections = from.connect(to, weight);
        for (let i = 0; i < connections.length; i++) {
            let connection = connections[i];
            if (from !== to) {
                this.connections.push(connection);
            } else {
                this.selfconns.push(connection);
            }
        }
        return connections;
    }

    disconnect(from: any, to: any): void {
        let connections = from === to ? this.selfconns : this.connections;
        for (let i = 0; i < connections.length; i++) {
            let connection = connections[i];
            if (connection.from === from && connection.to === to) {
                if (connection.gater !== null) this.ungate(connection);
                connections.splice(i, 1);
                break;
            }
        }
        from.disconnect(to);
    }

    gate(node: any, connection: any): void {
        if (this.nodes.indexOf(node) === -1) {
            throw new Error('This node is not part of the network!');
        } else if (connection.gater != null) {
            if (config.warnings) console.warn('This connection is already gated!');
            return;
        }
        node.gate(connection);
        this.gates.push(connection);
    }

    ungate(connection: any): void {
        let index = this.gates.indexOf(connection);
        if (index === -1) {
            throw new Error('This connection is not gated!');
        }
        this.gates.splice(index, 1);
        connection.gater.ungate(connection);
    }

    remove(node: any): void {
        let index = this.nodes.indexOf(node);
        if (index === -1) {
            throw new Error('This node does not exist in the network!');
        }
        let gaters: any[] = [];
        this.disconnect(node, node);
        let inputs: any[] = [];
        for (let i = node.connections.in.length - 1; i >= 0; i--) {
            let connection = node.connections.in[i];
            if (mutation.SUB_NODE.keep_gates && connection.gater !== null && connection.gater !== node) {
                gaters.push(connection.gater);
            }
            inputs.push(connection.from);
            this.disconnect(connection.from, node);
        }
        let outputs: any[] = [];
        for (let i = node.connections.out.length - 1; i >= 0; i--) {
            let connection = node.connections.out[i];
            if (mutation.SUB_NODE.keep_gates && connection.gater !== null && connection.gater !== node) {
                gaters.push(connection.gater);
            }
            outputs.push(connection.to);
            this.disconnect(node, connection.to);
        }
        let connections: any[] = [];
        for (let i = 0; i < inputs.length; i++) {
            let input = inputs[i];
            for (let j = 0; j < outputs.length; j++) {
                let output = outputs[j];
                if (!input.isProjectingTo(output)) {
                    let conn = this.connect(input, output);
                    connections.push(conn[0]);
                }
            }
        }
        for (let i = 0; i < gaters.length; i++) {
            if (connections.length === 0) break;
            let gater = gaters[i];
            let connIndex = Math.floor(Math.random() * connections.length);
            this.gate(gater, connections[connIndex]);
            connections.splice(connIndex, 1);
        }
        for (let i = node.connections.gated.length - 1; i >= 0; i--) {
            let conn = node.connections.gated[i];
            this.ungate(conn);
        }
        this.disconnect(node, node);
        this.nodes.splice(index, 1);
    }

    mutate(method: any): void {
        if (typeof method === 'undefined') {
            throw new Error('No (correct) mutate method given!');
        }
        let i: number, j: number;
        switch (method) {
            case mutation.ADD_NODE: {
                const connection = this.connections[Math.floor(Math.random() * this.connections.length)];
                const gater = connection.gater;
                this.disconnect(connection.from, connection.to);
                const toIndex = this.nodes.indexOf(connection.to);
                const node = new Node('hidden');
                node.mutate(mutation.MOD_ACTIVATION);
                const minBound = Math.min(toIndex, this.nodes.length - this.output);
                this.nodes.splice(minBound, 0, node);
                const newConn1 = this.connect(connection.from, node)[0];
                const newConn2 = this.connect(node, connection.to)[0];
                if (gater != null) {
                    this.gate(gater, Math.random() >= 0.5 ? newConn1 : newConn2);
                }
                break;
            }
            case mutation.SUB_NODE: {
                if (this.nodes.length === this.input + this.output) {
                    if (config.warnings) console.warn('No more nodes left to remove!');
                    break;
                }
                const index = Math.floor(Math.random() * (this.nodes.length - this.output - this.input) + this.input);
                this.remove(this.nodes[index]);
                break;
            }
            case mutation.ADD_CONN: {
                const available: any[] = [];
                for (i = 0; i < this.nodes.length - this.output; i++) {
                    let node1 = this.nodes[i];
                    for (j = Math.max(i + 1, this.input); j < this.nodes.length; j++) {
                        let node2 = this.nodes[j];
                        if (!node1.isProjectingTo(node2)) available.push([node1, node2]);
                    }
                }
                if (available.length === 0) {
                    if (config.warnings) console.warn('No more connections to be made!');
                    break;
                }
                const pair = available[Math.floor(Math.random() * available.length)];
                this.connect(pair[0], pair[1]);
                break;
            }
            case mutation.SUB_CONN: {
                const possible: any[] = [];
                for (i = 0; i < this.connections.length; i++) {
                    let conn = this.connections[i];
                    if (conn.from.connections.out.length > 1 && conn.to.connections.in.length > 1 && this.nodes.indexOf(conn.to) > this.nodes.indexOf(conn.from)) {
                        possible.push(conn);
                    }
                }
                if (possible.length === 0) {
                    if (config.warnings) console.warn('No connections to remove!');
                    break;
                }
                const randomConn = possible[Math.floor(Math.random() * possible.length)];
                this.disconnect(randomConn.from, randomConn.to);
                break;
            }
            case mutation.MOD_WEIGHT: {
                const allConnections = this.connections.concat(this.selfconns);
                const selectedConnection = allConnections[Math.floor(Math.random() * allConnections.length)];
                const modification = Math.random() * (method.max - method.min) + method.min;
                selectedConnection.weight += modification;
                break;
            }
            case mutation.MOD_BIAS: {
                const biasIndex = Math.floor(Math.random() * (this.nodes.length - this.input) + this.input);
                const biasNode = this.nodes[biasIndex];
                biasNode.mutate(method);
                break;
            }
            case mutation.MOD_ACTIVATION: {
                if (!method.mutateOutput && this.input + this.output === this.nodes.length) {
                    if (config.warnings) console.warn('No nodes that allow mutation of activation function');
                    break;
                }
                const activationIndex = Math.floor(Math.random() * (this.nodes.length - (method.mutateOutput ? 0 : this.output) - this.input) + this.input);
                const activationNode = this.nodes[activationIndex];
                activationNode.mutate(method);
                break;
            }
            case mutation.ADD_SELF_CONN: {
                const selfPossible: any[] = [];
                for (i = this.input; i < this.nodes.length; i++) {
                    let node = this.nodes[i];
                    if (node.connections.self.weight === 0) {
                        selfPossible.push(node);
                    }
                }
                if (selfPossible.length === 0) {
                    if (config.warnings) console.warn('No more self-connections to add!');
                    break;
                }
                const selfNode = selfPossible[Math.floor(Math.random() * selfPossible.length)];
                this.connect(selfNode, selfNode);
                break;
            }
            case mutation.SUB_SELF_CONN: {
                if (this.selfconns.length === 0) {
                    if (config.warnings) console.warn('No more self-connections to remove!');
                    break;
                }
                const selfConn = this.selfconns[Math.floor(Math.random() * this.selfconns.length)];
                this.disconnect(selfConn.from, selfConn.to);
                break;
            }
            case mutation.ADD_GATE: {
                const allConns = this.connections.concat(this.selfconns);
                const gatePossible: any[] = [];
                for (i = 0; i < allConns.length; i++) {
                    let conn = allConns[i];
                    if (conn.gater === null) {
                        gatePossible.push(conn);
                    }
                }
                if (gatePossible.length === 0) {
                    if (config.warnings) console.warn('No more connections to gate!');
                    break;
                }
                const gateIndex = Math.floor(Math.random() * (this.nodes.length - this.input) + this.input);
                const gateNode = this.nodes[gateIndex];
                const gateConn = gatePossible[Math.floor(Math.random() * gatePossible.length)];
                this.gate(gateNode, gateConn);
                break;
            }
            case mutation.SUB_GATE: {
                if (this.gates.length === 0) {
                    if (config.warnings) console.warn('No more connections to ungate!');
                    break;
                }
                const ungateIndex = Math.floor(Math.random() * this.gates.length);
                const gatedConn = this.gates[ungateIndex];
                this.ungate(gatedConn);
                break;
            }
            case mutation.ADD_BACK_CONN: {
                const backAvailable: any[] = [];
                for (i = this.input; i < this.nodes.length; i++) {
                    let node1 = this.nodes[i];
                    for (j = this.input; j < i; j++) {
                        let node2 = this.nodes[j];
                        if (!node1.isProjectingTo(node2)) backAvailable.push([node1, node2]);
                    }
                }
                if (backAvailable.length === 0) {
                    if (config.warnings) console.warn('No more connections to be made!');
                    break;
                }
                const backPair = backAvailable[Math.floor(Math.random() * backAvailable.length)];
                this.connect(backPair[0], backPair[1]);
                break;
            }
            case mutation.SUB_BACK_CONN: {
                const backPossible: any[] = [];
                for (i = 0; i < this.connections.length; i++) {
                    let conn = this.connections[i];
                    if (conn.from.connections.out.length > 1 && conn.to.connections.in.length > 1 && this.nodes.indexOf(conn.from) > this.nodes.indexOf(conn.to)) {
                        backPossible.push(conn);
                    }
                }
                if (backPossible.length === 0) {
                    if (config.warnings) console.warn('No connections to remove!');
                    break;
                }
                const randomBackConn = backPossible[Math.floor(Math.random() * backPossible.length)];
                this.disconnect(randomBackConn.from, randomBackConn.to);
                break;
            }
            case mutation.SWAP_NODES: {
                if ((method.mutateOutput && this.nodes.length - this.input < 2) || (!method.mutateOutput && this.nodes.length - this.input - this.output < 2)) {
                    if (config.warnings) console.warn('No nodes that allow swapping of bias and activation function');
                    break;
                }
                const swapIndex1 = Math.floor(Math.random() * (this.nodes.length - (method.mutateOutput ? 0 : this.output) - this.input) + this.input);
                const node1 = this.nodes[swapIndex1];
                const swapIndex2 = Math.floor(Math.random() * (this.nodes.length - (method.mutateOutput ? 0 : this.output) - this.input) + this.input);
                const node2 = this.nodes[swapIndex2];
                [node1.bias, node2.bias] = [node2.bias, node1.bias];
                [node1.squash, node2.squash] = [node2.squash, node1.squash];
                break;
            }
        }
    }

    /**
     * Create an offspring from two parent networks by crossing over their genes.
     * @param network1 The first parent network
     * @param network2 The second parent network
     * @param equal If true, treat both parents as equal fitness
     * @returns The offspring network
     */
    static crossOver(network1: Network, network2: Network, equal: boolean = false): Network {
        if (network1.input !== network2.input || network1.output !== network2.output) {
            throw new Error("Networks don't have the same input/output size!");
        }

        // Initialise offspring
        const offspring = new Network(network1.input, network1.output);
        offspring.connections = [];
        offspring.nodes = [];

        // Save scores and create a copy
        const score1 = (network1 as any).score || 0;
        const score2 = (network2 as any).score || 0;

        // Determine offspring node size
        let size: number;
        if (equal || score1 === score2) {
            const max = Math.max(network1.nodes.length, network2.nodes.length);
            const min = Math.min(network1.nodes.length, network2.nodes.length);
            size = Math.floor(Math.random() * (max - min + 1) + min);
        } else if (score1 > score2) {
            size = network1.nodes.length;
        } else {
            size = network2.nodes.length;
        }

        const outputSize = network1.output;

        // Set indexes for nodes
        for (let i = 0; i < network1.nodes.length; i++) {
            network1.nodes[i].index = i;
        }
        for (let i = 0; i < network2.nodes.length; i++) {
            network2.nodes[i].index = i;
        }

        // Assign nodes from parents to offspring
        for (let i = 0; i < size; i++) {
            let node: any;
            if (i < size - outputSize) {
                const random = Math.random();
                node = random >= 0.5 ? network1.nodes[i] : network2.nodes[i];
                const other = random < 0.5 ? network1.nodes[i] : network2.nodes[i];
                if (typeof node === 'undefined' || node.type === 'output') {
                    node = other;
                }
            } else {
                if (Math.random() >= 0.5) {
                    node = network1.nodes[network1.nodes.length + i - size];
                } else {
                    node = network2.nodes[network2.nodes.length + i - size];
                }
            }
            const newNode = new (Object.getPrototypeOf(node).constructor)();
            newNode.bias = node.bias;
            newNode.squash = node.squash;
            newNode.type = node.type;
            offspring.nodes.push(newNode);
        }

        // Create arrays of connection genes
        const n1conns: Record<string, any> = {};
        const n2conns: Record<string, any> = {};
        const Connection = network1.connections[0]?.constructor || network2.connections[0]?.constructor;

        // Normal and self connections for network1
        for (const conn of network1.connections.concat(network1.selfconns)) {
            const data = {
                weight: conn.weight,
                from: conn.from.index,
                to: conn.to.index,
                gater: conn.gater != null ? conn.gater.index : -1
            };
            n1conns[Connection.innovationID(data.from, data.to)] = data;
        }
        // Normal and self connections for network2
        for (const conn of network2.connections.concat(network2.selfconns)) {
            const data = {
                weight: conn.weight,
                from: conn.from.index,
                to: conn.to.index,
                gater: conn.gater != null ? conn.gater.index : -1
            };
            n2conns[Connection.innovationID(data.from, data.to)] = data;
        }

        // Split common conn genes from disjoint or excess conn genes
        const connections: any[] = [];
        const keys1 = Object.keys(n1conns);
        const keys2 = Object.keys(n2conns);
        for (let i = keys1.length - 1; i >= 0; i--) {
            // Common gene
            if (typeof n2conns[keys1[i]] !== 'undefined') {
                const conn = Math.random() >= 0.5 ? n1conns[keys1[i]] : n2conns[keys1[i]];
                connections.push(conn);
                n2conns[keys1[i]] = undefined;
            } else if (score1 >= score2 || equal) {
                connections.push(n1conns[keys1[i]]);
            }
        }
        // Excess/disjoint gene
        if (score2 >= score1 || equal) {
            for (let i = 0; i < keys2.length; i++) {
                if (typeof n2conns[keys2[i]] !== 'undefined') {
                    connections.push(n2conns[keys2[i]]);
                }
            }
        }

        // Add common conn genes uniformly
        for (const connData of connections) {
            if (connData.to < size && connData.from < size) {
                const from = offspring.nodes[connData.from];
                const to = offspring.nodes[connData.to];
                const conn = offspring.connect(from, to)[0];
                conn.weight = connData.weight;
                if (connData.gater !== -1 && connData.gater < size) {
                    offspring.gate(offspring.nodes[connData.gater], conn);
                }
            }
        }
        return offspring;
    }

    /**
     * Merge two networks into one. Output of network1 must match input of network2.
     * @param network1 The first network (output)
     * @param network2 The second network (input)
     * @returns The merged network
     */
    static merge(network1: Network, network2: Network): Network {
        // Create a copy of the networks
        network1 = Network.fromJSON(network1.toJSON());
        network2 = Network.fromJSON(network2.toJSON());

        // Check if output and input size are the same
        if (network1.output !== network2.input) {
            throw new Error('Output size of network1 should be the same as the input size of network2!');
        }

        // Redirect all connections from network2 input from network1 output
        for (let i = 0; i < network2.connections.length; i++) {
            const conn = network2.connections[i];
            if (conn.from.type === 'input') {
                const index = network2.nodes.indexOf(conn.from);
                // redirect
                conn.from = network1.nodes[network1.nodes.length - 1 - index];
            }
        }

        // Delete input nodes of network2
        for (let i = network2.input - 1; i >= 0; i--) {
            network2.nodes.splice(i, 1);
        }

        // Change the node type of network1's output nodes (now hidden)
        for (let i = network1.nodes.length - network1.output; i < network1.nodes.length; i++) {
            network1.nodes[i].type = 'hidden';
        }

        // Create one network from both networks
        network1.connections = network1.connections.concat(network2.connections);
        network1.nodes = network1.nodes.concat(network2.nodes);

        return network1;
    }

    /**
     * Convert the network to a json object
     */
    public toJSON(): any {
        let connections: any[] = [];
        for (let i = 0; i < this.connections.length; i++) {
            let connection = this.connections[i];
            connections.push({
                from: connection.from.index,
                to: connection.to.index,
                weight: connection.weight,
                gater: connection.gater !== null ? connection.gater.index : null
            });
        }
        let nodes: any[] = [];
        for (let i = 0; i < this.nodes.length; i++) {
            let node = this.nodes[i];
            nodes.push({
                bias: node.bias,
                squash: node.squash,
                type: node.type,
                index: i
            });
        }
        let json = {
            nodes: nodes,
            connections: connections,
            input: this.input,
            output: this.output,
            dropout: this.dropout
        };
        return json;
    }

    /**
     * Convert a json object to a network
     */
    public static fromJSON(json: any): Network {
        let network = new Network(json.input, json.output);
        network.dropout = json.dropout;
        network.nodes = [];
        network.connections = [];

        // Create node instances
        for (let i = 0; i < json.nodes.length; i++) {
            let nodeData = json.nodes[i];
            let node = new Node(nodeData.type);
            node.bias = nodeData.bias;
            node.squash = nodeData.squash;
            node.index = i;
            network.nodes.push(node);
        }

        // Create connections
        for (let i = 0; i < json.connections.length; i++) {
            let connData = json.connections[i];
            let fromNode = network.nodes[connData.from];
            let toNode = network.nodes[connData.to];
            let gaterNode = connData.gater !== null ? network.nodes[connData.gater] : null;
            network.connect(fromNode, toNode, connData.weight);
            if (gaterNode !== null) {
                network.gate(gaterNode, network.connections[network.connections.length - 1]);
            }
        }

        return network;
    }

    /**
     * Test the network on a dataset and return error
     * @param set Array of {input, output}
     * @param costFn Cost function
     */
    public test(set: { input: number[]; output: number[] }[], costFn: (target: number[], output: number[]) => number): { error: number } {
        let error = 0;
        for (let i = 0; i < set.length; i++) {
            const input = set[i].input;
            const target = set[i].output;
            const output = this.activate(input);
            error += costFn(target, output);
        }
        return { error: error / set.length };
    }

    // ...existing code for train, test, and other methods...
}

export default Network;
