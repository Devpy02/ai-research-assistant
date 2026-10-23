# Documentación de demostración: servicio local

Ollama permite ejecutar modelos de lenguaje en el equipo local y expone una API HTTP en la dirección http://localhost:11434.

El endpoint /api/generate recibe solicitudes de generación de texto. El endpoint /api/embed permite calcular representaciones vectoriales de textos.

Un fragmento documental debe conservar el identificador del documento de origen, el nombre del archivo, el índice del fragmento y sus offsets de caracteres. Esta trazabilidad permite localizar posteriormente la evidencia utilizada por el asistente.
