from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retrival_script import retriever

model = OllamaLLM(model="qwen2.5-coder:3b")

template = """
You are an expert AI system specialized in **code translation**, specifically converting **C++ code to Python**.

### **Task:**
- Take **pure C++ code** as input.
- Use additional context retrieved from the **vector database** to improve accuracy.
- Generate **optimized and idiomatic** Python code that preserves the original logic and best practices.

### **Additional Context:**
{context}

### **C++ Code to Convert:**
{cpp_code}

### **Expected Output:**
Provide the equivalent **Python code**.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


print("\n\n-------------------------------")
cpp_code = """
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h> 

static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
   int i;
   for(i = 0; i<argc; i++) {
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
}

int main(int argc, char* argv[]) {
   sqlite3 *db;
   char *zErrMsg = 0;
   int rc;
   char *sql;

   /* Open database */
   rc = sqlite3_open("test.db", &db);
   
   if( rc ) {
      fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
      return(0);
   } else {
      fprintf(stdout, "Opened database successfully\n");
   }

   /* Create SQL statement */
   sql = "CREATE TABLE COMPANY("  \
      "ID INT PRIMARY KEY     NOT NULL," \
      "NAME           TEXT    NOT NULL," \
      "AGE            INT     NOT NULL," \
      "ADDRESS        CHAR(50)," \
      "SALARY         REAL );";

   /* Execute SQL statement */
   rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
   
   if( rc != SQLITE_OK ){
      fprintf(stderr, "SQL error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
   } else {
      fprintf(stdout, "Table created successfully\n");
   }
   sqlite3_close(db);
   return 0;
}"""
print("\n\n")
print("🔍 **C++ Code to Convert:**\n")
print(cpp_code)
print("\n\n")

#context = "---"
context = retriever.invoke(cpp_code)
print("🔍 **Context Retrieved from Vector Database:**\n")
print(context)
result = chain.invoke({"context": context, "cpp_code": cpp_code})
print("🟢 **Generated Python Code:**\n")
print(result)
