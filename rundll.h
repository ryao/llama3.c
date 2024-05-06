#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Main Main;

Main *build_main(char* checkpoint_path, char* tokenizer_path, float temperature, float topp, int steps,
                 char* prompt, unsigned long long rng_seed, char* mode, char* system_prompt);
void free_main(Main *m);
int run_main(Main *m);

#ifdef __cplusplus
}
#endif

#endif // MAIN_H
