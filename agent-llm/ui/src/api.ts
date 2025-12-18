import axios from 'axios';

const API_BASE = '/api';

export interface TestStep {
    test_step: string;
    expected_result: string;
    note_to_llm?: string;
    skipped?: boolean;
}

export interface SavedTest {
    name: string;
    filename: string;
    steps_count: number;
    action_id: string;
    modified_at: number;
}

export interface RunRequest {
    scenario_name: string;
    steps: TestStep[];
    temperature?: number;
}

export interface StepOutcome {
    step: {
        test_step: string;
        expected_result: string;
        note_to_llm?: string;
    };
    result: {
        status: string;
        attempts: number;
        reason?: string;
        actions: any[];
    };
}

export interface ScenarioResult {
    status: string;
    steps: StepOutcome[];
    final_state?: any;
    reason?: string;
    _meta?: {
        duration_sec: number;
        total_steps: number;
        passed_steps: number;
        failed_steps: number;
        total_attempts: number;
        backend_config: {
            provider: string;
            model: string;
            max_tokens: number;
        };
    };
}

export interface ConfigResponse {
    status: string;
    provider: string;
    model: string;
    action_url: string;
}

const api = {
    // Health check
    async healthCheck(): Promise<boolean> {
        try {
            const resp = await axios.get(`${API_BASE}/healthz`);
            return resp.data.status === 'ok';
        } catch {
            return false;
        }
    },

    // Get config
    async getConfig(): Promise<ConfigResponse> {
        const resp = await axios.get(`${API_BASE}/config`);
        return resp.data;
    },

    // List saved tests
    async listTests(): Promise<SavedTest[]> {
        const resp = await axios.get(`${API_BASE}/list-tests`);
        return resp.data.tests || [];
    },

    // Run new scenario with LLM
    async runScenario(request: RunRequest): Promise<ScenarioResult> {
        const resp = await axios.post(`${API_BASE}/run`, request);
        return resp.data;
    },

    // Replay saved test
    async replayTest(testName: string): Promise<any> {
        const resp = await axios.post(`${API_BASE}/run-saved-test`, { test_name: testName });
        return resp.data;
    },

    // Delete test
    async deleteTest(testName: string): Promise<void> {
        await axios.delete(`${API_BASE}/delete-test/${encodeURIComponent(testName)}`);
    },

    // Get test details
    async getTestDetails(testName: string): Promise<any> {
        const resp = await axios.get(`${API_BASE}/get-test/${encodeURIComponent(testName)}`);
        return resp.data;
    },

    // Stop execution
    async stopExecution(): Promise<void> {
        await axios.post(`${API_BASE}/stop`);
    }
};

export default api;

