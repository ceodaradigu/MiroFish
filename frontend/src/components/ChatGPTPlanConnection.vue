<template>
  <section v-if="planMode !== false" class="plan-connection" aria-labelledby="plan-connection-title">
    <div class="card-heading">
      <h3 id="plan-connection-title">{{ copy.title }}</h3>
      <span class="connection-state" :class="{ ready: connectionReady }" role="status">
        {{ stateLabel }}
      </span>
    </div>

    <p>{{ copy.description }}</p>
    <p class="consent-note">{{ copy.consent }}</p>

    <div class="connection-actions">
      <a
        class="connect-button"
        :href="startUrl"
        target="_blank"
        rel="noopener noreferrer"
        data-oai-siwc-button
      >{{ status.connected ? copy.reconnect : copy.connect }}</a>
      <button type="button" class="refresh-button" :disabled="loadingStatus" @click="refreshStatus">
        {{ copy.refresh }}
      </button>
    </div>

    <p v-if="status.connected && !status.planUsageEnabled" class="attention" role="status">
      {{ copy.notEligible }}
    </p>

    <div v-if="status.planUsageEnabled" class="model-picker">
      <p class="plan-usage-label">
        {{ copy.usingPlan }}
        <a href="https://chatgpt.com/settings/usage" target="_blank" rel="noopener noreferrer">{{ copy.manageUsage }}</a>
      </p>
      <label for="chatgpt-plan-model">{{ copy.modelLabel }}</label>
      <div class="model-controls">
        <select id="chatgpt-plan-model" v-model="pendingModel" :disabled="loadingModels || savingModel || models.length === 0">
          <option value="" disabled>{{ copy.chooseModel }}</option>
          <option v-for="model in models" :key="model.slug" :value="model.slug">
            {{ model.display_name || model.slug }}
          </option>
        </select>
        <button type="button" :disabled="!pendingModel || savingModel || pendingModel === status.selectedModel" @click="saveModel">
          {{ savingModel ? copy.saving : copy.saveModel }}
        </button>
      </div>
      <p v-if="!loadingModels && models.length === 0" class="attention">{{ copy.noModels }}</p>
    </div>

    <p class="zep-note" :class="{ attention: !status.memoryReady }">
      {{ status.memoryBackend === 'local' ? copy.localMemory : (status.zepConfigured ? copy.zepReady : copy.zepRequired) }}
    </p>
    <p v-if="status.memoryBackend === 'local'" class="consent-note">{{ copy.localPrivacy }}</p>
    <p v-if="errorMessage" class="attention" role="alert">{{ errorMessage }}</p>
  </section>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import api from '../api'

const emit = defineEmits(['status-change', 'memory-backend-change'])
const { locale } = useI18n()
const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'
const startUrl = `${apiBase}/api/chatgpt/start`

const messages = {
  es: {
    title: 'Conectar ChatGPT',
    description: 'Autoriza esta instalación local de MiroFish para usar tu plan de ChatGPT. La disponibilidad se comprueba al iniciar sesión y depende de tu cuenta y plan.',
    consent: 'Al continuar, aceptarás el acceso en la página oficial de OpenAI. La conexión puede usar los límites de tu plan. No introduzcas credenciales ni claves aquí.',
    connect: 'Continuar con ChatGPT',
    reconnect: 'Reconectar cuenta',
    refresh: 'Comprobar estado',
    checking: 'Comprobando…',
    disconnected: 'Sin conectar',
    connected: 'Conectado',
    unavailable: 'Acceso por plan no habilitado',
    notEligible: 'La cuenta conectada no tiene habilitado este acceso. Comprueba la respuesta de OpenAI al iniciar sesión.',
    modelLabel: 'Modelo autorizado',
    usingPlan: 'Usando plan ChatGPT ·',
    manageUsage: 'Gestionar uso',
    chooseModel: 'Selecciona un modelo',
    saveModel: 'Guardar modelo',
    saving: 'Guardando…',
    noModels: 'No hay modelos disponibles para esta cuenta.',
    zepRequired: 'Zep Cloud sigue siendo necesario para las simulaciones y se configura por separado.',
    zepReady: 'Zep Cloud está configurado por separado.',
    localMemory: 'Memoria local seleccionada: los grafos se guardan en este ordenador, sin cuenta de Zep.',
    localPrivacy: 'La extracción usa ChatGPT y la base local no está cifrada. Usa solo datos públicos o ficticios.',
    statusError: 'No se pudo comprobar la conexión local. Verifica que el servidor esté iniciado.',
    modelsError: 'No se pudo cargar la lista de modelos autorizados.',
    saveError: 'No se pudo guardar el modelo.'
  },
  en: {
    title: 'Connect ChatGPT',
    description: 'Authorize this local MiroFish installation to use your ChatGPT plan. Availability is checked during sign-in and depends on your account and plan.',
    consent: 'Continuing opens the official OpenAI authorization page. Usage may count toward your plan limits. Do not enter credentials or keys here.',
    connect: 'Continue with ChatGPT',
    reconnect: 'Reconnect account',
    refresh: 'Check status',
    checking: 'Checking…',
    disconnected: 'Not connected',
    connected: 'Connected',
    unavailable: 'Plan access unavailable',
    notEligible: 'This account does not have plan access enabled. Check the OpenAI response during sign-in.',
    modelLabel: 'Authorized model',
    usingPlan: 'Using ChatGPT plan ·',
    manageUsage: 'Manage usage',
    chooseModel: 'Select a model',
    saveModel: 'Save model',
    saving: 'Saving…',
    noModels: 'No models are available for this account.',
    zepRequired: 'Zep Cloud is still required for simulations and is configured separately.',
    zepReady: 'Zep Cloud is configured separately.',
    localMemory: 'Local memory selected: graphs stay on this computer; no Zep account is needed.',
    localPrivacy: 'Extraction uses ChatGPT and the local database is not encrypted. Use public or fictional data only.',
    statusError: 'Could not check the local connection. Make sure the server is running.',
    modelsError: 'Could not load the authorized models.',
    saveError: 'Could not save the model.'
  },
  zh: {
    title: '连接 ChatGPT',
    description: '授权本地 MiroFish 使用你的 ChatGPT 套餐。登录时会检查账户和套餐是否支持。',
    consent: '继续后将打开 OpenAI 官方授权页面。使用量可能计入套餐限额。请勿在此输入凭据或密钥。',
    connect: '使用 ChatGPT 继续',
    reconnect: '重新连接账户',
    refresh: '检查状态',
    checking: '检查中…',
    disconnected: '未连接',
    connected: '已连接',
    unavailable: '套餐访问不可用',
    notEligible: '此账户未启用套餐访问。请查看登录时 OpenAI 的提示。',
    modelLabel: '已授权模型',
    usingPlan: '正在使用 ChatGPT 套餐 ·',
    manageUsage: '管理用量',
    chooseModel: '选择模型',
    saveModel: '保存模型',
    saving: '保存中…',
    noModels: '此账户暂无可用模型。',
    zepRequired: '模拟仍需要单独配置 Zep Cloud。',
    zepReady: 'Zep Cloud 已单独配置。',
    localMemory: '已选择本地记忆：图谱保存在此电脑，无需 Zep 账户。',
    localPrivacy: '提取仍使用 ChatGPT，本地数据库未加密。请仅使用公开或虚构数据。',
    statusError: '无法检查本地连接，请确认服务器已启动。',
    modelsError: '无法加载已授权模型。',
    saveError: '无法保存模型。'
  }
}

const copy = computed(() => {
  if (navigator.language?.toLowerCase().startsWith('es')) return messages.es
  return messages[locale.value] || messages.en
})
const status = reactive({ connected: false, planUsageEnabled: false, memoryBackend: 'zep', memoryReady: false, zepConfigured: false, selectedModel: '' })
const planMode = ref(null)
const models = ref([])
const pendingModel = ref('')
const loadingStatus = ref(true)
const loadingModels = ref(false)
const savingModel = ref(false)
const errorMessage = ref('')
const connectionReady = computed(() => planMode.value === false || (
  planMode.value === true && status.connected && status.planUsageEnabled && status.memoryReady &&
  !!status.selectedModel && models.value.some(model => model.slug === status.selectedModel)
))
const stateLabel = computed(() => {
  if (loadingStatus.value) return copy.value.checking
  if (!status.connected) return copy.value.disconnected
  return status.planUsageEnabled ? copy.value.connected : copy.value.unavailable
})

function notifyReadiness() {
  emit('status-change', connectionReady.value)
}

async function loadModels() {
  loadingModels.value = true
  try {
    const response = await api.get('/api/chatgpt/models')
    models.value = Array.isArray(response.models)
      ? response.models.filter(model => model && typeof model.slug === 'string' && model.slug)
      : []
  } catch {
    models.value = []
    errorMessage.value = copy.value.modelsError
  } finally {
    loadingModels.value = false
  }
}

async function refreshStatus() {
  loadingStatus.value = true
  errorMessage.value = ''
  models.value = []
  notifyReadiness()
  try {
    const response = await api.get('/api/chatgpt/status')
    planMode.value = response.planMode === false ? false : true
    status.connected = response.connected === true
    status.planUsageEnabled = response.planUsageEnabled === true
    status.memoryBackend = response.memoryBackend === 'local' ? 'local' : 'zep'
    status.memoryReady = response.memoryReady === true
    status.zepConfigured = response.zepConfigured === true
    status.selectedModel = typeof response.selectedModel === 'string' ? response.selectedModel : ''
    pendingModel.value = status.selectedModel
    if (planMode.value && status.connected && status.planUsageEnabled) await loadModels()
    else models.value = []
  } catch {
    planMode.value = null
    status.connected = false
    status.planUsageEnabled = false
    status.memoryBackend = 'zep'
    status.memoryReady = false
    status.zepConfigured = false
    status.selectedModel = ''
    models.value = []
    pendingModel.value = ''
    errorMessage.value = copy.value.statusError
  } finally {
    loadingStatus.value = false
    emit('memory-backend-change', planMode.value === true ? status.memoryBackend : null)
    notifyReadiness()
  }
}

async function saveModel() {
  if (!pendingModel.value || !models.value.some(model => model.slug === pendingModel.value)) return
  savingModel.value = true
  errorMessage.value = ''
  try {
    await api.post('/api/chatgpt/model', { model: pendingModel.value })
    await refreshStatus()
  } catch {
    errorMessage.value = copy.value.saveError
  } finally {
    savingModel.value = false
  }
}

onMounted(() => {
  refreshStatus()
  window.addEventListener('focus', refreshStatus)
})
onUnmounted(() => window.removeEventListener('focus', refreshStatus))
</script>

<style scoped>
.plan-connection { border: 1px solid #d6d6d6; padding: 20px; margin: 12px 0 24px; font-size: .9rem; line-height: 1.5; }
.card-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
h3 { margin: 0; font-size: 1.15rem; }
p { margin: 8px 0; }
.connection-state { border: 1px solid #bbb; padding: 3px 7px; font-size: .72rem; white-space: nowrap; }
.connection-state.ready { border-color: #277e46; color: #176132; }
.consent-note, .zep-note { color: #555; }
.connection-actions, .model-controls { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.connect-button, button, select { font: inherit; }
.connect-button, button { padding: 8px 10px; border: 1px solid #222; text-decoration: none; cursor: pointer; }
.connect-button { background: #111; color: #fff; }
button { background: #fff; color: #111; }
button:disabled { opacity: .5; cursor: not-allowed; }
.model-picker { margin-top: 18px; }
.plan-usage-label { font-size: .8rem; font-weight: 600; }
.plan-usage-label a { color: #184d9e; }
.model-picker label { display: block; font-weight: 600; }
select { flex: 1; min-width: 160px; padding: 8px; border: 1px solid #999; background: #fff; }
.attention { color: #9a3108; }
@media (max-width: 650px) { .card-heading { align-items: flex-start; flex-direction: column; } }
</style>
