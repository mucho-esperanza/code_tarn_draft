from src.code_parser import CodeParser
from src.prompt_engineer import PromptEngineer
from src.model_client import ModelClient
from src.output_processor import OutputProcessor
from src.code_translator import CodeTranslator

def simple_example():
    """Example usage of the code translator with direct API calls."""
    # Define the code sample
    code_sample = """
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h> 

static int callback(void *data, int argc, char **argv, char **azColName){
   int i;
   fprintf(stderr, "%s: ", (const char*)data);
   
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
   const char* data = "Callback function called";

   /* Open database */
   rc = sqlite3_open("test.db", &db);
   
   if( rc ) {
      fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
      return(0);
   } else {
      fprintf(stderr, "Opened database successfully\n");
   }

   /* Create merged SQL statement */
   sql = "UPDATE COMPANY set SALARY = 25000.00 where ID=1; " \
         "SELECT * from COMPANY";

   /* Execute SQL statement */
   rc = sqlite3_exec(db, sql, callback, (void*)data, &zErrMsg);
   
   if( rc != SQLITE_OK ) {
      fprintf(stderr, "SQL error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
   } else {
      fprintf(stdout, "Operation done successfully\n");
   }
   sqlite3_close(db);
   return 0;
}
    """
    
    # Create a translator instance
    model_client = ModelClient(model="codellama:latest", temperature=0.3)
    translator = CodeTranslator("C++", "Python", model_client)
    
    # Translate the code
    results = translator.translate(code_sample)
    
    # Display results
    if results["success"]:
        print("Translation successful!")
        print("\nCode Structure:")
        for key, value in results["code_structure"].items():
            print(f"  {key}: {value}")
            
        print("\nTranslated Code:")
        print(results["formatted_code"])
        
        print(f"\nSyntax Valid: {results['syntax_valid']}")
    else:
        print(f"Translation failed: {results['error']}")

if __name__ == "__main__":
    simple_example()