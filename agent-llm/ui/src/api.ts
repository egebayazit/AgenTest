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
    mode?: 'test' | 'agent';
}

export interface RunRequest {
    scenario_name: string;
    steps: TestStep[];
    temperature?: number;
}

export interface TokenUsage {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    provider: string;
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
        actions: {
            action_id: string;
            plan: any;
            ack: any;
            state_before: any;
            state_after: any;
            token_usage?: TokenUsage;
        }[];
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

export interface LLMSettings {
    provider: string;
    base_url: string;
    model: string;
    api_key_masked?: string;
    has_api_key?: boolean;
}

export interface LLMSettingsInput {
    provider: string;
    base_url: string;
    model: string;
    api_key?: string;
}

export interface LLMProvider {
    id: string;
    name: string;
    requires_key: boolean;
}

// ============================================================================
// AGENT MODE TYPES
// ============================================================================

export interface AgentStep {
    instruction: string;
    note?: string;
}

export interface AgentRunRequest {
    scenario_name: string;
    instructions: AgentStep[];
    temperature?: number;
    use_ods?: boolean;
}

export interface AgentStepOutcome {
    instruction: {
        instruction: string;
        note?: string;
    };
    result: {
        status: string;
        actions: {
            action_id: string;
            plan: any;
            ack: any;
            state_before: any;
            state_after: any;
            token_usage?: TokenUsage;
        }[];
        final_state: any;
        last_plan: any;
        reason?: string;
    };
}

export interface AgentResult {
    status: string;
    steps: AgentStepOutcome[];
    final_state?: any;
    reason?: string;
    _meta?: {
        mode: string;
        duration_sec: number;
        total_instructions: number;
        executed_count: number;
        failed_count: number;
        provider: string;
        use_ods: boolean;
    };
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
    },

    // ============================================================================
    // SETTINGS API
    // ============================================================================

    // Get current LLM settings
    async getSettings(): Promise<LLMSettings> {
        const resp = await axios.get(`${API_BASE}/settings`);
        return resp.data.settings;
    },

    // Save LLM settings
    async saveSettings(settings: LLMSettingsInput): Promise<void> {
        await axios.post(`${API_BASE}/settings`, settings);
    },

    // Test LLM connection
    async testConnection(): Promise<{ success: boolean; message: string }> {
        const resp = await axios.post(`${API_BASE}/settings/test`);
        return resp.data;
    },

    // Get available providers
    async getProviders(): Promise<LLMProvider[]> {
        const resp = await axios.get(`${API_BASE}/settings/providers`);
        return resp.data.providers;
    },

    // ============================================================================
    // AGENT MODE API
    // ============================================================================

    // Run agent scenario (no semantic filtering, no pre-check, no expected result)
    async runAgentScenario(request: AgentRunRequest): Promise<AgentResult> {
        const resp = await axios.post(`${API_BASE}/run-agent`, request);
        return resp.data;
    }
}

export default api;

