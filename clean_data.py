import csv

input_file = "nedrex-edge-20250218.csv"
output_file = "nedrex-edge-20250218-ok.csv"

# Abrir el archivo de entrada y salida
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8", newline="") as outfile:
    # Crear un lector y escritor CSV
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    buffer = ""
    
    for line in reader:
        # Unir todas las columnas de la línea para manejar posibles saltos de línea dentro de un campo
        buffer += ','.join(line)
        
        # Contar las comillas dobles para saber si estamos dentro de un campo con comillas
        quote_count = buffer.count('"')
        
        if quote_count % 2 == 0:
            # Reemplazar los saltos de línea internos por un espacio
            buffer = buffer.replace('"', ' "').replace('\r\n', ' ').replace('\n', ' ')
            
            # Escribir la línea procesada
            writer.writerow([buffer])
            
            # Resetear el buffer
            buffer = ""

print(f"Proceso completado. Archivo guardado como: {output_file}")
