typedef struct frame {
  si key, val;
  struct frame *next;
} frame;

void init_frames(void);

// SORT
si put(si fm, si key, si val);
///
